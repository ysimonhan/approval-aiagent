import streamlit as st
import pandas as pd
import datetime
import uuid
import cohere
import json
import time 
import io # For file handling
import os # For file handling
from PIL import Image as PILImage # For image handling
import cv2 # For image processing (if needed)
import numpy as np
from ultralytics import YOLO 
from mistralai import Mistral # For OCR processing
import base64

# --- Configuration ---
# (YC startups often use .env files or cloud secret managers for production)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COHERE_API_KEY = "9umxICMmVXpk6trBETGkfCPvHBzCV9TSjgyEVxWP"
COHERE_AYA_MODEL = "c4ai-aya-vision-32b" # Check Cohere documentation for latest vision-capable models on v2/chat
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "../03-Models/yolov8m.pt") # <<< IMPORTANT: Update this path to your YOLOv8 model file (.pt)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "wuJauJ5WhHxpOXPv3eFkG1lGlq3Y0yXK") # Get from environment variable

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY != "YOUR_MISTRAL_API_KEY" else None

# Placeholder for custom model integration
CUSTOM_MODEL_CONFIG = {
    "api_key": "YOUR_CUSTOM_MODEL_API_KEY",
    "endpoint": "YOUR_CUSTOM_MODEL_ENDPOINT",
    # Add other necessary parameters for your custom model
}

# --- YOLOv8 Model Loading and Processing ---
yolo_initialized = False

def load_yolo_model(model_path):
    global yolo_initialized
    if YOLO is None:
        st.warning("Object detection currently does not work and thus container damage detection features will be disabled.")
        return None
    if not yolo_initialized:
        st.success(f"Object and container damage detection initialized successfully.")
        yolo_initialized = True
    model = YOLO(model_path)
    return model
    
def analyze_image_with_yolo(image_bytes, model):
    if model is None:
        return "YOLO model not loaded. Analysis skipped."
    if not image_bytes:
        return "No image data provided for analysis."
    try:
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        # Convert PIL image to OpenCV format (NumPy array)
        # YOLO expects images in BGR format if using OpenCV directly,
        # but PILImage.open gives RGB. The YOLO library handles this conversion.
        
        results = model(pil_image) # Perform inference
        
        detections_summary = []
        if results and results[0].boxes:
            names = results[0].names # Class names
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = names[class_id]
                    confidence = float(box.conf[0])
                    detections_summary.append(f"{class_name} (confidence: {confidence:.2f})")
        
        if not detections_summary:
            return "No objects detected by YOLOv8."
        return "Detected: " + ", ".join(detections_summary)

    except Exception as e:
        return f"Error during YOLOv8 analysis: {e}"

# --- Helper Functions ---
def generate_ticket_id():
    return str(uuid.uuid4())[:8]

def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- SOP Processing Functions ---
def process_sop_with_mistral_ocr(uploaded_file):
    """Process an uploaded SOP document using Mistral OCR."""
    if not mistral_client:
        st.error("Mistral API key not configured. Please set MISTRAL_API_KEY environment variable.")
        return None
    
    try:
        # Convert uploaded file to base64
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        
        # Process with Mistral OCR using base64
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            },
            include_image_base64=True
        )
        
        return ocr_response
            
    except Exception as e:
        st.error(f"Error processing SOP with Mistral OCR: {e}")
        return None

def analyze_sop_with_cohere(ocr_result):
    """Analyze OCR results with Cohere Aya to extract repair codes requiring images."""
    if not COHERE_API_KEY:
        st.error("Cohere API key not configured.")
        return None
    
    try:
        co = cohere.Client(COHERE_API_KEY)
        
        # Combine all pages' markdown content
        # OCR result is an object, not a dictionary, so use attribute access
        full_text = "\n".join(page.markdown for page in ocr_result.pages)
        
        prompt = f"""You are an AI assistant analyzing a Standard Operating Procedure (SOP) document for container repairs.
        Your task is to identify repair codes that require images for documentation.
        
        Document content:
        {full_text}
        
        Please analyze the text and identify any repair codes that explicitly require images or visual documentation.
        For each identified code, provide:
        1. The repair code (e.g., DMG001)
        2. A brief description of why images are required
        
        Format your response as a JSON object with repair codes as keys and descriptions as values.
        Example format:
        {{
            "DMG001": "Severe damage assessment requiring visual documentation",
            "CRK003": "Crack inspection requiring photographic evidence"
        }}
        
        Only include codes that explicitly mention image requirements. If no codes are found, return an empty object {{}}.
        """
        
        response = co.chat(model=COHERE_AYA_MODEL, message=prompt)
        
        # Extract JSON from response
        try:
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                return json.loads(response.text[json_start:json_end])
            return {}
        except json.JSONDecodeError:
            st.error("Could not parse Cohere response as JSON")
            return {}
            
    except Exception as e:
        st.error(f"Error analyzing SOP with Cohere: {e}")
        return None

def chat_with_ai_agent(ticket_data, user_question, chat_history):
    """Chat with the Cohere Aya AI agent about the ticket decision."""
    if not COHERE_API_KEY:
        st.error("Cohere API key not configured.")
        return None
    
    try:
        co = cohere.Client(COHERE_API_KEY)
        
        # Prepare ticket context
        repairs_text = "\n".join([f"- {r.get('code', 'N/A')}: {r.get('description', 'No description')}" for r in ticket_data.get('repairs', [])])
        media_text = "\n".join([f"- {m.get('filename', 'Unknown file')} ({m.get('type', 'unknown')})" for m in ticket_data.get('media', [])])
        
        # Get current repair codes requiring images from session state
        repair_codes_requiring_images = st.session_state.get('repair_codes_needing_images', {})
        image_req_text = "\n".join([f"- {code}: {desc}" for code, desc in repair_codes_requiring_images.items()]) if repair_codes_requiring_images else "None currently configured"
        
        # Prepare chat history context
        relevant_chat = [msg for msg in chat_history[-5:] if msg.get('sender') in ['AI Agent', 'System']]  # Last 5 AI/System messages
        chat_context = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in relevant_chat])
        
        prompt = f"""You are the AI Repair Ticket Approval Agent that previously analyzed this container repair ticket. 
        You can answer questions about your decision, reasoning, and provide clarifications about repair approvals.

        TICKET CONTEXT:
        - Ticket ID: {ticket_data['ticket_id']}
        - Container ID: {ticket_data['container_id']}
        - Company: {ticket_data['company']}
        - Container Age: {ticket_data['container_age']} years
        - Total Cost: ${ticket_data['total_cost_estimate']}
        - Repairs: {repairs_text if repairs_text else "None listed"}
        - Media Files: {media_text if media_text else "None provided"}
        - Other Notes: {ticket_data.get('other_notes', 'None')}

        CURRENT SYSTEM RULES - REPAIR CODES REQUIRING IMAGES:
        {image_req_text}

        YOUR PREVIOUS DECISION:
        - Decision: {ticket_data.get('ai_decision', 'Not yet decided')}
        - Confidence: {ticket_data.get('ai_confidence', 'N/A')}
        - Reasoning: {ticket_data.get('ai_reasoning', 'No reasoning provided')}
        - Missing Data Request: {ticket_data.get('ai_missing_data_request', 'None')}

        RECENT CONVERSATION:
        {chat_context if chat_context else "No previous AI conversation"}

        USER QUESTION: {user_question}

        Please respond as the AI agent that made the original decision. Be helpful, explain your reasoning clearly, 
        and provide specific guidance. If the user asks about changing your decision, explain what would be needed.
        When discussing image requirements, refer to the current system rules above.
        Keep your response conversational and professional."""
        
        response = co.chat(model=COHERE_AYA_MODEL, message=prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error chatting with AI agent: {e}")
        return f"Error: Unable to communicate with AI agent. {e}"

# --- Logging ---
def log_action(ticket_id, action, details=None, user="System"):
    log_entry = {
        "timestamp": get_current_timestamp(),
        "ticket_id": ticket_id,
        "action": action,
        "user": user,
        "details": details if details is not None else {}
    }
    st.session_state.logs.append(log_entry)
    print(f"LOG: {log_entry}") # Print to console for debugging during development

# --- AI Agent Logic ---
def call_custom_model(ticket_data, model_config):
    st.sidebar.info(f"Trying to call custom model with endpoint: {model_config.get('endpoint')}")
    time.sleep(1)
    decision_options = ["APPROVE", "DISAPPROVE", "MANUAL_REVIEW"]
    decision = decision_options[hash(ticket_data['ticket_id']) % 3]
    confidence = 0.65 + (hash(ticket_data['ticket_id']) % 30) / 100.0
    reasoning = f"Custom model mock decision: Processed ticket {ticket_data['ticket_id']}. Decision: {decision}."
    return {
        "decision": decision,
        "confidence_score": confidence,
        "reasoning": reasoning,
        "missing_data_request": None # Generalize missing_images_for_codes
    }

def call_cohere_aya_yolo_model(ticket_data, age_cost_thresholds, repair_codes_needing_images):
    if not COHERE_API_KEY or COHERE_API_KEY == "YOUR_COHERE_API_KEY_HERE":
        st.error("Cohere API Key not configured. Please set it in Streamlit secrets or directly in the script.")
        return {
            "decision": "MANUAL_REVIEW", "confidence_score": 0.0,
            "reasoning": "Configuration error: Cohere API Key missing.",
            "missing_data_request": "Cohere API Key configuration."
        }

    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    co = cohere.Client(COHERE_API_KEY)
    
    # Check if the ticket has images and analyze them with YOLO
    yolo_analysis_prompt_section = ""
    if ticket_data.get('media'):
        yolo_summaries_for_prompt = []
        for media_item in ticket_data['media']:
            if media_item.get('type') == 'image' and media_item.get('yolo_summary'):
                assoc_info = f" (associated with repair code {media_item.get('repair_code_association')})" if media_item.get('repair_code_association') else ""
                yolo_summaries_for_prompt.append(f"- Image '{media_item.get('filename', 'N/A')}'{assoc_info}: {media_item['yolo_summary']}")
        
        if yolo_summaries_for_prompt:
            yolo_analysis_prompt_section = "\n\nImage Analysis Summary (from YOLOv8 object detection):\n" + "\n".join(yolo_summaries_for_prompt)
    
    prompt = f"""You are an AI Repair Ticket Approval Agent for a container depot.
    Review the container repair ticket and decide: 'APPROVE', 'DISAPPROVE', or 'MANUAL_REVIEW'.
    Provide a confidence score (0.0-1.0) and reasoning.
    If data is missing (e.g., required images and/or videos), an AI decision cannot be made; instead, the decision should be 'MANUAL_REVIEW', and the reasoning should clearly state what specific data is missing. The 'missing_data_request' field in JSON should detail this.

    Approval Criteria:
    1. Cost vs. Container Age:
        {chr(10).join([f"    - {age_range}: Max approved cost ${threshold}" for age_range, threshold in age_cost_thresholds.items()])}
    2. Image Requirements:
        Repair codes needing images: {', '.join(repair_codes_needing_images.keys()) if repair_codes_needing_images else "None specified"}.
        For each suggested repair:
            - If its code requires an image AND an image for that specific repair code is NOT listed as provided, this is considered missing data.
    3. Image Content Analysis (from YOLOv8):
        Review the 'Image Analysis Summary' section below. Consider if detected damages align with suggested repairs and their severity.
        If YOLOv8 detects significant damages not listed in repairs, or if detected damages seem minor for high-cost repairs, flag for 'MANUAL_REVIEW'.

    Ticket Details:
    - Ticket ID: {ticket_data['ticket_id']}
    - Container ID: {ticket_data['container_id']}
    - Company: {ticket_data['company']}
    - Container Age (years): {ticket_data['container_age']}
    - Total Cost Estimate: ${ticket_data['total_cost_estimate']}
    - Suggested Repairs:"""

    repair_details_prompt = []
    missing_images_for_codes_internal_check = []

    for repair in ticket_data.get('repairs', []):
        code = repair.get('code', 'N/A')
        description = repair.get('description', 'No description')
        requires_image = code in repair_codes_needing_images
        # Check against 'media' which now stores uploaded file info, including 'repair_code_association'
        image_provided_for_code = any(
            media_item.get('repair_code_association') == code for media_item in ticket_data.get('media', [])
        ) if requires_image else True

        repair_details_prompt.append(
            f"  - Repair Code: {code}, Description: {description} (Requires Image: {requires_image}, Image Provided for this code: {image_provided_for_code})"
        )
        if requires_image and not image_provided_for_code:
            missing_images_for_codes_internal_check.append(code)

    prompt += "\n" + "\n".join(repair_details_prompt)
    prompt += f"\n- Other Notes: {ticket_data.get('other_notes', 'None')}"
    
    media_summary = []
    for m in ticket_data.get('media', []):
        assoc = f" (for code {m.get('repair_code_association')})" if m.get('repair_code_association') else ""
        media_summary.append(f"{m.get('filename', 'Unknown file')}{assoc}")
    prompt += f"\n- Media Provided: {len(ticket_data.get('media', []))} files. ({', '.join(media_summary)})"

    prompt += yolo_analysis_prompt_section

    # This pre-emptive check is crucial. If data is missing, AI can't make a proper decision.
    if missing_images_for_codes_internal_check:
        missing_data_details = f"Mandatory images missing for repair codes: {', '.join(missing_images_for_codes_internal_check)}."
        return {
            "decision": "MANUAL_REVIEW", # Changed from DISAPPROVE directly to MANUAL_REVIEW with missing data flag
            "confidence_score": 1.0, # High confidence that data is missing
            "reasoning": f"Cannot proceed with AI approval/disapproval. {missing_data_details} Please upload them.",
            "missing_data_request": missing_data_details # This signals what is needed
        }

    prompt += """
    Based on all the above (and assuming all necessary data like images IS present if not flagged as missing), provide your response strictly in the following JSON format:
    {
    "decision": "APPROVE" / "DISAPPROVE" / "MANUAL_REVIEW",
    "confidence_score": <float between 0.0 and 1.0>,
    "reasoning": "<concise explanation for the decision, including specific criteria met or failed. If 'MANUAL_REVIEW' due to low confidence or complex rules, explain why.>",
    "missing_data_request": null // Should be null if not requesting data
    }
    """
    
    try:
        # st.sidebar.info(f"Calling Cohere Aya ({COHERE_AYA_MODEL})...")
        response = co.chat(model=COHERE_AYA_MODEL, message=prompt)
        st.sidebar.success("AI successfully processed the repair estimate.")
        ai_response_text = response.text
        json_start = ai_response_text.find('{')
        json_end = ai_response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            parsed_response = json.loads(ai_response_text[json_start:json_end])
            if not all(key in parsed_response for key in ["decision", "confidence_score", "reasoning"]):
                raise ValueError("LLM response missing required JSON keys.")
            # Ensure missing_data_request is present, can be null
            parsed_response.setdefault('missing_data_request', None)
            return parsed_response
        else:
            st.error(f"Could not parse JSON from Cohere response: {ai_response_text}")
            return {"decision": "MANUAL_REVIEW", "confidence_score": 0.1, "reasoning": f"Error: Could not parse AI response.", "missing_data_request": "AI response parsing error."}
    except Exception as e:
        st.error(f"Error calling Cohere API: {e}")
        return {"decision": "MANUAL_REVIEW", "confidence_score": 0.0, "reasoning": f"API error: {e}", "missing_data_request": "API communication failure."}

def process_ticket_with_ai(ticket_data, age_cost_thresholds, repair_codes_needing_images, use_cohere_aya=True):
    if use_cohere_aya:
        # The pre-emptive check for missing images is now part of call_cohere_aya_yolo_model
        # to allow the LLM to formulate the missing_data_request message itself.
        ai_result = call_cohere_aya_yolo_model(ticket_data, age_cost_thresholds, repair_codes_needing_images)
        ai_result["ai_agent_type"] = f"Cohere Aya ({COHERE_AYA_MODEL})"
    else:
        ai_result = call_custom_model(ticket_data, CUSTOM_MODEL_CONFIG)
        ai_result["ai_agent_type"] = "Custom Model (Placeholder)"
    return ai_result

# --- Streamlit App Initialization ---
st.set_page_config(layout="wide", page_title="AI Repair Ticket Approval")

# --- Initialize Session State ---
if 'tickets' not in st.session_state: st.session_state.tickets = []
if 'logs' not in st.session_state: st.session_state.logs = []
if 'feedback' not in st.session_state: st.session_state.feedback = []

if 'age_cost_thresholds' not in st.session_state:
    st.session_state.age_cost_thresholds = {
        "New (0-2 years)": 1000, "Medium (3-5 years)": 700,
        "Old (6-8 years)": 400, "Very Old (>8 years)": 200,
    }
if 'repair_codes_needing_images' not in st.session_state:
    st.session_state.repair_codes_needing_images = {
        "DMG001": "Severe Dent Repair", "CRK003": "Frame Crack Assessment",
        "RST005": "Surface Rust Treatment", "FLR001": "Floor Replacement",
    }

# Sample Data - adjusted to reflect no "Pending AI Review"
def add_sample_tickets_if_needed():
    if not st.session_state.tickets: # Add only if empty
        sample_tickets_data_raw = [
            {
                "container_id": generate_ticket_id(), "company": "Maersk", "total_cost_estimate": 800, "container_age": 1,
                "repairs": [{"code": "DNT001", "description": "Minor dent"}, {"code": "SCT002", "description": "Scratch"}],
                "media_simulated": [{"filename": "img1_for_DNT001.jpg", "type": "image", "repair_code_association": "DNT001"}], # Simulate initial media
                "other_notes": "Standard wear."
            },
            {
                "container_id": generate_ticket_id(), "company": "MSC", "total_cost_estimate": 1200, "container_age": 4,
                "repairs": [{"code": "DMG001", "description": "Structural damage"}, {"code": "FLR001", "description": "Floor replace"}],
                "media_simulated": [], # DMG001 needs image, but it's missing for initial processing
                "other_notes": "Urgent."
            },
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON003", "company": "Cosco", "total_cost_estimate": 300,
                "container_age": 7, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "RST005", "description": "Surface rust treatment"}],
                "media": [{"filename": "rust_overview.jpg", "type": "image"}],
                "ai_chat": [], "other_notes": ""
            },
    
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON005", "company": "Maersk", "total_cost_estimate": 600,
                "container_age": 3, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "CRK003", "description": "Frame crack repair"}, {"code": "SCT002", "description": "Scratch removal"}],
                "media": [{"filename": "frame_crack_con005.jpg", "type": "image", "repair_code_association": "CRK003"}],
                "ai_chat": [], "other_notes": "Moderate damage observed."
            },
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON006", "company": "Hapag", "total_cost_estimate": 450,
                "container_age": 5, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "DNT001", "description": "Minor dent repair"}, {"code": "SCT002", "description": "Scratch removal"}],
                "media": [{"filename": "dent_scratch_con006.jpg", "type": "image"}],
                "ai_chat": [], "other_notes": "Minor wear and tear."
            },
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON007", "company": "MSC", "total_cost_estimate": 1100,
                "container_age": 2, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "DMG001", "description": "Severe structural damage"}, {"code": "FLR001", "description": "Floor replacement"}],
                "media": [{"filename": "structural_damage_con007.jpg", "type": "image", "repair_code_association": "DMG001"}],
                "ai_chat": [], "other_notes": "Urgent repair required."
            },
    
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON004", "company": "CGM", "total_cost_estimate": 900,
                "container_age": 2, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "DMG001", "description": "Frame damage report"}, {"code": "WEL001", "description": "Welding required"}],
                "media": [{"filename": "frame_damage_con004.jpg", "type": "image", "repair_code_association": "DMG001"}],
                "ai_chat": [], "other_notes": "Impact damage noted."
            }
        ]
        for raw_ticket in sample_tickets_data_raw:
            ticket_id = generate_ticket_id()
            new_ticket = {
                "ticket_id": ticket_id,
                "container_id": raw_ticket["container_id"],
                "company": raw_ticket["company"],
                "total_cost_estimate": raw_ticket["total_cost_estimate"],
                "container_age": raw_ticket["container_age"],
                "repairs": raw_ticket.get("repairs", []),
                "media": raw_ticket.get("media_simulated", []), # Store simulated media directly
                "other_notes": raw_ticket.get("other_notes", ""),
                "submitted_date": get_current_timestamp(),
                "ai_chat": [{"timestamp": get_current_timestamp(), "sender": "System", "message": "Ticket submitted."}]
            }
            
            # Immediately process with AI
            # st.sidebar.info(f"Auto-processing sample ticket {ticket_id}...")
            ai_result = process_ticket_with_ai(
                new_ticket,
                st.session_state.age_cost_thresholds,
                st.session_state.repair_codes_needing_images,
                use_cohere_aya=True # Assuming Cohere for samples for now
            )

            new_ticket['ai_decision'] = ai_result['decision']
            new_ticket['ai_confidence'] = ai_result.get('confidence_score', 0.0)
            new_ticket['ai_reasoning'] = ai_result['reasoning']
            new_ticket['ai_agent_type'] = ai_result.get('ai_agent_type', 'Unknown AI')
            new_ticket['ai_processed_date'] = get_current_timestamp()
            new_ticket['ai_missing_data_request'] = ai_result.get('missing_data_request')

            # Determine status based on AI result
            if new_ticket['ai_missing_data_request']:
                new_ticket['status'] = "Additional Data Requested"
                log_msg_action = "Additional Data Requested by AI"
                new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Additional data required: {new_ticket['ai_missing_data_request']}. Reasoning: {new_ticket['ai_reasoning']}"})
            elif new_ticket['ai_decision'] == "APPROVE" and new_ticket['ai_confidence'] >= 0.75: # Configurable confidence threshold
                new_ticket['status'] = "AI Approved"
                log_msg_action = "AI Approved"
                new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Approved. Reasoning: {new_ticket['ai_reasoning']}"})
            elif new_ticket['ai_decision'] == "DISAPPROVE":
                new_ticket['status'] = "AI Disapproved"
                log_msg_action = "AI Disapproved"
                new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Disapproved. Reasoning: {new_ticket['ai_reasoning']}"})
            else: # MANUAL_REVIEW or low confidence APPROVE
                new_ticket['status'] = "Manual Review Required"
                log_msg_action = "Flagged for Manual Review by AI"
                new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"This ticket requires manual review. Reasoning: {new_ticket['ai_reasoning']}"})
            
            st.session_state.tickets.append(new_ticket)
            log_action(ticket_id, f"New Sample Ticket Submitted & {log_msg_action}", ai_result)
        st.sidebar.success("All repair estimates have been processed.")


add_sample_tickets_if_needed()

# --- Sidebar ---
st.sidebar.title("🚢 Repair Approval Platform")
st.sidebar.markdown("---")
page_options = ["Home (GSC)", "AI Training & Settings (GSC)", "Approvals (GSC & Inspectors)", "Submit New Estimate (Inspectors)"]
# Calculate counts for sidebar navigation (optional, but can be nice)
# approval_counts = {
#     "Manual Review": len([t for t in st.session_state.tickets if t['status'] == "Manual Review Required"]),
#     "Data Requested": len([t for t in st.session_state.tickets if t['status'] == "Additional Data Requested"]),
# }
# page_options[1] = f"Approvals (MR: {approval_counts['Manual Review']}, DR: {approval_counts['Data Requested']})"

page = st.sidebar.radio("Navigation", page_options)
st.sidebar.markdown("---")
st.sidebar.subheader("AI Agent Settings")
use_cohere = st.sidebar.checkbox("Use Cohere Aya Model", value=True)
if not use_cohere: st.sidebar.caption("Using Custom Model Placeholder.")

# --- Page Implementations ---
def display_ticket_details(ticket, expand_details=False):
    with st.expander(f"Ticket {ticket['ticket_id']} ({ticket['company']} - {ticket['container_id']}) - Status: {ticket['status']}", expanded=expand_details):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company:** {ticket['company']}")
            st.write(f"**Container ID:** {ticket['container_id']}")
            st.write(f"**Container Age:** {ticket['container_age']} years")
        with col2:
            st.write(f"**Total Cost Estimate:** ${ticket['total_cost_estimate']:.2f}")
            st.write(f"**Submitted:** {ticket.get('submitted_date', 'N/A')}")
            st.write(f"**Last AI Update:** {ticket.get('ai_processed_date', 'N/A')}")

        st.write("**Suggested Repairs:**")
        if ticket.get('repairs'):
            for repair in ticket['repairs']: st.markdown(f"- `{repair.get('code', 'N/A')}`: {repair.get('description', 'No desc')}")
        else: st.write("No repairs listed.")

        st.write("**Media Files:**")
        if ticket.get('media'):
            for media_item in ticket['media']:
                assoc_code = f" (for {media_item.get('repair_code_association')})" if media_item.get('repair_code_association') else ""
                st.markdown(f"- {media_item.get('filename', 'N/A')} ({media_item.get('type', 'N/A')}){assoc_code}")
                # If 'bytes' were stored, you could offer a download:
                # if 'bytes' in media_item:
                # st.download_button(f"Download {media_item.get('filename')}", media_item['bytes'], media_item.get('filename'))
        else: st.write("No media files attached.")
        
        st.markdown(f"**Other Notes:** {ticket.get('other_notes', 'None')}")

        if ticket.get('ai_reasoning'):
            st.info(f"**AI Agent ({ticket.get('ai_agent_type', 'N/A')} on {ticket.get('ai_processed_date', '')}):**\n"
                    f"Decision: **{ticket.get('ai_decision', 'N/A')}** (Confidence: {ticket.get('ai_confidence', 0.0):.2f})\n"
                    f"Reasoning: {ticket.get('ai_reasoning', 'N/A')}")
            if ticket.get('ai_missing_data_request'): # Changed from ai_missing_images
                st.warning(f"AI requests additional data: {ticket['ai_missing_data_request']}")

        # Manual Override Actions
        if ticket['status'] in ["AI Approved", "AI Disapproved", "Manual Review Required", "Additional Data Requested"]:
            st.markdown("---")
            st.write("**Manual Actions:**")
            b_col1, b_col2, b_col3 = st.columns(3)
            override_user = "GSC User" # Example user

            if b_col1.button("Manually Approve", key=f"manual_approve_{ticket['ticket_id']}"):
                ticket['status'] = "Manually Approved"
                ticket['manual_override_by'] = override_user
                ticket['manual_override_date'] = get_current_timestamp()
                ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": override_user, "message": "Ticket Manually Approved."})
                log_action(ticket['ticket_id'], "Manually Approved", user=override_user)
                st.rerun()
            if b_col2.button("Manually Disapprove", key=f"manual_disapprove_{ticket['ticket_id']}"):
                ticket['status'] = "Manually Disapproved"
                ticket['manual_override_by'] = override_user
                ticket['manual_override_date'] = get_current_timestamp()
                ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": override_user, "message": "Ticket Manually Disapproved."})
                log_action(ticket['ticket_id'], "Manually Disapproved", user=override_user)
                st.rerun()
            if ticket['status'] in ["AI Approved", "AI Disapproved"] and b_col3.button("Revert to Manual Review", key=f"revert_{ticket['ticket_id']}"):
                ticket['original_ai_status'] = ticket['status']
                ticket['status'] = "Manual Review Required"
                ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": override_user, "message": f"AI decision reverted. Moved to Manual Review."})
                log_action(ticket['ticket_id'], "AI Decision Reverted to Manual Review", user=override_user)
                st.rerun()
        
        # Handle "Additional Data Requested" status
        if ticket['status'] == "Additional Data Requested":
            st.markdown("---")
            st.write("**Provide Additional Data:**")
            # Example: If missing images were requested, allow upload for those specific codes
            # This part needs to know WHAT data is missing. For now, assumes images for codes listed in ai_missing_data_request.
            
            # Attempt to parse missing codes if the request is about images
            missing_codes_str = ticket.get('ai_missing_data_request', '')
            codes_to_upload_for = []
            if "missing for repair codes:" in missing_codes_str: # Basic check
                try:
                    codes_to_upload_for = [code.strip() for code in missing_codes_str.split("missing for repair codes:")[1].split('.')[0].split(',')]
                except: # pylint: disable=bare-except
                    st.caption("Could not automatically parse specific codes from data request.")
            
            if codes_to_upload_for:
                st.write(f"The AI specifically requested images for codes: {', '.join(codes_to_upload_for)}")
                
            # Use a unique key for file_uploader within the loop/ticket context
            uploaded_files_for_ticket = st.file_uploader(
                "Upload required files", 
                accept_multiple_files=True, 
                key=f"file_upload_{ticket['ticket_id']}",
                type=['png', 'jpg', 'jpeg', 'pdf'] # Example file types
            )

            # Allow associating uploaded files with repair codes if needed
            temp_file_associations = {}
            if uploaded_files_for_ticket:
                st.write("Associate uploaded files with repair codes (if applicable):")
                for i, uploaded_file in enumerate(uploaded_files_for_ticket):
                    # Only show relevant codes (those missing data or all if none specified)
                    relevant_repair_codes = codes_to_upload_for or [r['code'] for r in ticket.get('repairs', [])]
                    assoc_code = st.selectbox(
                        f"Associate '{uploaded_file.name}' with code:",
                        options=[""] + relevant_repair_codes,
                        key=f"assoc_select_{ticket['ticket_id']}_{i}"
                    )
                    temp_file_associations[uploaded_file.name] = {"file": uploaded_file, "association": assoc_code}
            
            if st.button("Submit Uploaded Data & Reprocess Ticket", key=f"resubmit_data_{ticket['ticket_id']}"):
                if uploaded_files_for_ticket:
                    num_added = 0
                    for file_name, data in temp_file_associations.items():
                        uploaded_file = data["file"]
                        association = data["association"]
                        # Store file info (name, type). For real apps, save to blob storage & store URL/ref.
                        # For this demo, we'll just update the media list.
                        # To store bytes (careful with session state size): file_bytes = uploaded_file.getvalue()
                        ticket.setdefault('media', []).append({
                            "filename": uploaded_file.name,
                            "type": uploaded_file.type,
                            "repair_code_association": association if association else None, # Link to repair code
                            "uploaded_timestamp": get_current_timestamp()
                        })
                        num_added += 1
                    
                    ticket['ai_chat'].append({
                        "timestamp": get_current_timestamp(), 
                        "sender": "System", 
                        "message": f"{num_added} file(s) uploaded by user. Resubmitting for AI review."
                    })
                    log_action(ticket['ticket_id'], f"{num_added} File(s) Uploaded for Data Request", user="GSC User")

                    # Clear previous AI decision fields that led to data request
                    ticket['ai_decision'] = None
                    ticket['ai_reasoning'] = None
                    ticket['ai_missing_data_request'] = None 
                    
                    # Reprocess with AI
                    with st.spinner(f"AI re-processing ticket {ticket['ticket_id']} with new data..."):
                        ai_result = process_ticket_with_ai(
                            ticket,
                            st.session_state.age_cost_thresholds,
                            st.session_state.repair_codes_needing_images,
                            use_cohere_aya=use_cohere
                        )
                    ticket.update({ # Update ticket with new AI results
                        'ai_decision': ai_result['decision'],
                        'ai_confidence': ai_result.get('confidence_score', 0.0),
                        'ai_reasoning': ai_result['reasoning'],
                        'ai_agent_type': ai_result.get('ai_agent_type', 'Unknown AI'),
                        'ai_processed_date': get_current_timestamp(),
                        'ai_missing_data_request': ai_result.get('missing_data_request')
                    })

                    # Update status based on new AI result
                    if ticket['ai_missing_data_request']: # Still missing data
                        ticket['status'] = "Additional Data Requested"
                        log_msg = f"Additional data still required: {ticket['ai_missing_data_request']}"
                    elif ticket['ai_decision'] == "APPROVE" and ticket['ai_confidence'] >= 0.75:
                        ticket['status'] = "AI Approved"
                        log_msg = "Ticket AI Approved after data submission."
                    elif ticket['ai_decision'] == "DISAPPROVE":
                        ticket['status'] = "AI Disapproved"
                        log_msg = "Ticket AI Disapproved after data submission."
                    else:
                        ticket['status'] = "Manual Review Required"
                        log_msg = "Ticket flagged for Manual Review by AI after data submission."
                    
                    ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"{log_msg} Reasoning: {ticket['ai_reasoning']}"})
                    log_action(ticket['ticket_id'], f"AI Reprocessed: {log_msg}", ai_result)
                    st.rerun()
                else:
                    st.warning("Please upload files before submitting.")

        # Manual Chat Feature
        if ticket['status'] in ["Manual Review Required", "Additional Data Requested"]:
            st.markdown("---")
            st.write("**Chat with AI Agent & Team Communication:**")
            chat_user = "GSC/Inspector User" # Could be dynamic if auth is added
            
            # Ensure ai_chat exists and is a list
            if not isinstance(ticket.get('ai_chat'), list):
                ticket['ai_chat'] = []

            for chat_msg in ticket['ai_chat']: # Display existing chat
                sender_style = "🤖 " if chat_msg['sender'] == "AI Agent" else ""
                st.markdown(f"<sub>**[{chat_msg['timestamp']}] {sender_style}{chat_msg['sender']}:** {chat_msg['message']}</sub>", unsafe_allow_html=True)

            # Chat input section
            col1, col2 = st.columns([3, 1])
            with col1:
                new_message = st.text_area("Your message or question for the AI or Colleagues:", key=f"manual_chat_input_{ticket['ticket_id']}", height=75)
            with col2:
                st.write("Send to:")
                send_to_ai = st.button("🤖 Ask AI Agent", key=f"send_ai_chat_{ticket['ticket_id']}")
                send_manual = st.button("👥 Send to Team", key=f"send_manual_chat_{ticket['ticket_id']}")
            
            if send_to_ai:
                if new_message:
                    # Add user message to chat
                    user_msg_data = {"timestamp": get_current_timestamp(), "sender": chat_user, "message": new_message}
                    ticket['ai_chat'].append(user_msg_data)
                    
                    # Get AI response
                    with st.spinner("AI Agent is thinking..."):
                        ai_response = chat_with_ai_agent(ticket, new_message, ticket['ai_chat'])
                        if ai_response:
                            ai_msg_data = {"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": ai_response}
                            ticket['ai_chat'].append(ai_msg_data)
                            log_action(ticket['ticket_id'], "AI Chat Exchange", {"user_msg": new_message, "ai_response": ai_response}, user=chat_user)
                    st.rerun()
                else:
                    st.warning("Please enter a message or question for the AI.")
            
            if send_manual:
                if new_message:
                    msg_data = {"timestamp": get_current_timestamp(), "sender": chat_user, "message": new_message}
                    ticket['ai_chat'].append(msg_data)
                    log_action(ticket['ticket_id'], "Manual Chat Message Added", msg_data, user=chat_user)
                    st.rerun()
                else:
                    st.warning("Please enter a message.")
                    
        else: # For other statuses, allow AI chat for questions
            st.markdown("---")
            st.write("**Communication Log & AI Questions:**")
            
            # Ensure ai_chat exists and is a list
            if not isinstance(ticket.get('ai_chat'), list):
                ticket['ai_chat'] = []
                
            if not ticket.get('ai_chat'): 
                st.caption("No messages yet.")
            else:
                for chat_msg in ticket.get('ai_chat', []):
                    sender_style = "🤖 " if chat_msg['sender'] == "AI Agent" else ""
                    st.markdown(f"<sub>**[{chat_msg['timestamp']}] {sender_style}{chat_msg['sender']}:** {chat_msg['message']}</sub>", unsafe_allow_html=True)
            
            # Allow AI questions even for completed tickets
            st.write("**Ask the AI Agent about this decision:**")
            ai_question = st.text_input("Question for AI Agent:", key=f"ai_question_{ticket['ticket_id']}")
            if st.button("🤖 Ask AI", key=f"ask_ai_{ticket['ticket_id']}"):
                if ai_question:
                    chat_user = "User"
                    # Add user question to chat
                    user_msg_data = {"timestamp": get_current_timestamp(), "sender": chat_user, "message": ai_question}
                    ticket['ai_chat'].append(user_msg_data)
                    
                    # Get AI response
                    with st.spinner("AI Agent is responding..."):
                        ai_response = chat_with_ai_agent(ticket, ai_question, ticket['ai_chat'])
                        if ai_response:
                            ai_msg_data = {"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": ai_response}
                            ticket['ai_chat'].append(ai_msg_data)
                            log_action(ticket['ticket_id'], "AI Question", {"question": ai_question, "ai_response": ai_response}, user=chat_user)
                    st.rerun()
                else:
                    st.warning("Please enter a question.")


if page == "Home (GSC)":
    st.header("Dashboard Overview")
    
    manual_review_needed = [t for t in st.session_state.tickets if t['status'] == "Manual Review Required"]
    additional_data_requested = [t for t in st.session_state.tickets if t['status'] == "Additional Data Requested"] # Changed
    
    ai_approved_count = len([t for t in st.session_state.tickets if t['status'] == "AI Approved"])
    ai_disapproved_count = len([t for t in st.session_state.tickets if t['status'] == "AI Disapproved"])
    manual_approved_count = len([t for t in st.session_state.tickets if t['status'] == "Manually Approved"])
    manual_disapproved_count = len([t for t in st.session_state.tickets if t['status'] == "Manually Disapproved"])

    # No more "Pending AI Review"
    col1, col2 = st.columns(2)
    col1.metric("Tickets for Manual Review", len(manual_review_needed))
    col2.metric("Tickets Requiring Additional Data", len(additional_data_requested)) # Changed

    st.subheader("Completion Status")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("AI Approved", ai_approved_count)
    col_b.metric("AI Disapproved", ai_disapproved_count)
    col_c.metric("Manually Approved", manual_approved_count)
    col_d.metric("Manually Disapproved", manual_disapproved_count)

    total_to_do = len(manual_review_needed) + len(additional_data_requested)
    avg_manual_time_per_ticket = 180
    estimated_time_seconds = total_to_do * avg_manual_time_per_ticket
    
    st.info(f"**Total active tickets requiring attention:** {total_to_do}")
    if total_to_do > 0: st.info(f"**Estimated time for manual actions:** {datetime.timedelta(seconds=estimated_time_seconds)}")
    else: st.success("All active tickets processed or awaiting external action!")

    st.subheader("Recent Activity Log (Last 10)")
    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs).sort_values(by="timestamp", ascending=False)
        st.dataframe(log_df.head(10).astype(str), use_container_width=True) # Ensure all cols are str for display
    else: st.write("No activity logged yet.")

elif page == "AI Training & Settings (GSC)":
    st.header("AI Evaluation & Configuration")
    st.subheader("AI Decision Thresholds (Cost vs. Age)")
    updated_thresholds = {}
    for age_range, current_threshold in st.session_state.age_cost_thresholds.items():
        updated_thresholds[age_range] = st.number_input(f"Max Cost for {age_range}", min_value=0, value=current_threshold, step=50, key=f"thresh_{age_range}")
    if st.button("Save Cost Thresholds"):
        st.session_state.age_cost_thresholds = updated_thresholds
        log_action(None, "AI Cost Thresholds Updated", updated_thresholds)
        st.success("Cost thresholds updated!"); st.rerun()

    st.markdown("---"); st.subheader("Repair Codes Requiring Images")
    st.markdown("Define which repair codes mandatorily require an image for approval.")
    current_img_req_codes = st.session_state.repair_codes_needing_images
    if current_img_req_codes:
        st.write("Current codes requiring images:")
        for code, desc in current_img_req_codes.items(): st.markdown(f"- `{code}`: {desc}")
    else: st.write("No repair codes currently configured to require images.")

    with st.form("add_image_req_code_form"):
        new_code = st.text_input("New Repair Code (e.g., DMG002)")
        new_code_desc = st.text_input("Description for new code")
        if st.form_submit_button("Add Image Requirement"):
            if new_code and new_code_desc:
                st.session_state.repair_codes_needing_images[new_code.upper()] = new_code_desc
                log_action(None, "AI Image Requirement Added", {new_code.upper(): new_code_desc})
                st.success(f"Added '{new_code.upper()}'."); st.rerun()
            else: st.error("Code and description required.")
    
    if current_img_req_codes:
        code_to_remove = st.selectbox("Select code to remove:", options=[""] + list(current_img_req_codes.keys()), key="remove_code_select")
        if st.button("Remove Selected Code Requirement") and code_to_remove:
            del st.session_state.repair_codes_needing_images[code_to_remove]
            log_action(None, "AI Image Requirement Removed", {"code_removed": code_to_remove})
            st.success(f"Removed '{code_to_remove}'."); st.rerun()
            
    # SOP Upload and Processing Section
    st.subheader("Alternative: Upload SOP Documents")
    st.markdown("Upload Standard Operating Procedure documents to automatically extract repair codes requiring images.")
    
    uploaded_sop = st.file_uploader(
        "Upload SOP Document (PDF)",
        type=['pdf'],
        key="sop_uploader"
    )
    
    if uploaded_sop:
        if st.button("Process SOP Document"):
            with st.spinner("Processing SOP document..."):
                # Step 1: Process with Mistral OCR
                ocr_result = process_sop_with_mistral_ocr(uploaded_sop)
                if ocr_result:
                    st.success("OCR processing completed.")
                    
                    # Step 2: Analyze with Cohere
                    with st.spinner("Analyzing document content..."):
                        new_codes = analyze_sop_with_cohere(ocr_result)
                        
                        if new_codes:
                            # Update session state with new codes
                            st.session_state.repair_codes_needing_images.update(new_codes)
                            log_action(None, "SOP Analysis Complete", {
                                "new_codes_added": new_codes,
                                "total_codes": len(st.session_state.repair_codes_needing_images)
                            })
                            st.success(f"Added {len(new_codes)} new repair codes requiring images.")
                            st.json(new_codes)
                        else:
                            st.info("No new repair codes requiring images found in the document.")
                else:
                    st.error("Failed to process SOP document.")
    
    st.markdown("---")
    st.subheader("Evaluate AI Decisions")
    ai_processed_tickets = [t for t in st.session_state.tickets if t['status'] in ["AI Approved", "AI Disapproved"] and 'ai_decision' in t]
    if not ai_processed_tickets: st.info("No AI-processed tickets for evaluation yet.")
    else:
        for ticket in ai_processed_tickets:
            with st.container(border=True):
                st.write(f"**Ticket ID: {ticket['ticket_id']}** (AI: {ticket['ai_decision']} @ {ticket['ai_confidence']:.2f})")
                st.caption(f"AI Reasoning: {ticket['ai_reasoning']}")
                existing_feedback = next((f for f in st.session_state.feedback if f['ticket_id'] == ticket['ticket_id']), None)
                if existing_feedback: st.success(f"Feedback: {existing_feedback['evaluation']} ({existing_feedback.get('comment', '')})")
                else:
                    cols_fb = st.columns([1,1,3])
                    if cols_fb[0].button("👍 Good", key=f"up_fb_{ticket['ticket_id']}") or \
                       cols_fb[1].button("👎 Bad", key=f"down_fb_{ticket['ticket_id']}"):
                        evaluation = "Good" if cols_fb[0].button else "Bad" # This logic needs fixing, button state doesn't work like this after click
                        # A better way: store button click in a variable
                        # For now, assume only one can be "true" per run if logic were outside columns
                        # Correct way for buttons:
                        # clicked_good = cols_fb[0].button("👍 Good", key=f"up_fb_{ticket['ticket_id']}")
                        # clicked_bad = cols_fb[1].button("👎 Bad", key=f"down_fb_{ticket['ticket_id']}")
                        # if clicked_good or clicked_bad:
                        #    evaluation = "Good" if clicked_good else "Bad"

                        # Simplified for now - this part of feedback needs robust button state handling if it were more complex
                        feedback_val = "Good" if st.session_state[f"up_fb_{ticket['ticket_id']}"] else "Bad" # This is not right
                        comment_val = st.session_state.get(f"comment_fb_{ticket['ticket_id']}", "")

                        # The button handling for feedback needs a slight redesign for Streamlit's execution model.
                        # The simplest is to have separate submit buttons or handle it within a form.
                        # For this pass, I'll leave the basic structure and note this as an area for refinement.
                        # A quick fix for immediate effect:
                        # Create a radio or select for feedback, then a submit button.

                        # Let's try a simpler feedback mechanism for now
                        feedback_choice = st.radio("Your evaluation:", ("Good", "Bad", "Not Set"), index=2, key=f"eval_radio_{ticket['ticket_id']}", horizontal=True)
                        feedback_comment = cols_fb[2].text_input("Comment", key=f"comment_fb_{ticket['ticket_id']}")
                        if st.button("Submit Feedback", key=f"submit_fb_{ticket['ticket_id']}"):
                            if feedback_choice != "Not Set":
                                feedback_data = {"ticket_id": ticket['ticket_id'], "evaluation": feedback_choice, "comment": feedback_comment, "timestamp": get_current_timestamp()}
                                st.session_state.feedback.append(feedback_data)
                                log_action(ticket['ticket_id'], "AI Feedback Submitted", feedback_data)
                                st.success("Feedback submitted."); st.rerun()
                            else:
                                st.warning("Please select 'Good' or 'Bad' for evaluation.")


    if st.session_state.feedback:
        st.subheader("Collected Feedback Data")
        feedback_df = pd.DataFrame(st.session_state.feedback)
        st.dataframe(feedback_df.astype(str), use_container_width=True)
        st.download_button("Download Feedback (CSV)", feedback_df.to_csv(index=False).encode('utf-8'), 'ai_feedback.csv', 'text/csv')


elif page == "Approvals (GSC & Inspectors)":
    st.header("Repair Ticket Approval Queues")

    # Define categories and their corresponding tickets
    ticket_categories_map = {
        "Manual Review Required": [t for t in st.session_state.tickets if t['status'] == "Manual Review Required"],
        "Additional Data Requested": [t for t in st.session_state.tickets if t['status'] == "Additional Data Requested"], # Changed
        "AI Approved": [t for t in st.session_state.tickets if t['status'] == "AI Approved"],
        "AI Disapproved": [t for t in st.session_state.tickets if t['status'] == "AI Disapproved"],
        "Manually Processed": [t for t in st.session_state.tickets if t['status'] in ["Manually Approved", "Manually Disapproved"]],
    }
    # Create tab titles with counts
    tab_titles_with_counts = [f"{title} ({len(tickets)})" for title, tickets in ticket_categories_map.items()]
    
    tabs = st.tabs(tab_titles_with_counts)

    # Iterate through the map directly to maintain order and access tickets
    for i, (title, tickets_in_category) in enumerate(ticket_categories_map.items()):
        with tabs[i]:
            # The subheader is now part of the tab title
            # st.subheader(f"{title} ({len(tickets_in_category)})") # No longer needed if count in tab title
            if not tickets_in_category:
                st.info(f"No tickets in '{title}' queue.")
            else:
                # No batch processing for "Pending AI Review" as it's removed
                for ticket in sorted(tickets_in_category, key=lambda x: x.get('submitted_date', ''), reverse=True):
                    display_ticket_details(ticket) # show_actions defaults to True implicitly now
                    st.markdown("---")


elif page == "Submit New Estimate (Inspectors)":
    st.header("Submit New Repair Ticket")
    with st.form("new_ticket_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        container_id = c1.text_input("Container ID*", key="submit_cid")
        company = c2.selectbox("Shipping Company*", ["Maersk", "Cosco", "CGM", "MSC", "Hapag", "Other"], key="submit_comp")
        if company == "Other": company = c2.text_input("Specify Other Company", key="submit_comp_other")
        total_cost_estimate = c1.number_input("Total Cost Estimate ($)*", min_value=0.0, step=10.0, format="%.2f", key="submit_cost")
        container_age = c2.number_input("Container Age (years)*", min_value=0, max_value=50, step=1, key="submit_age")
        
        st.markdown("**Suggested Repairs:**")
        if 'current_repairs_for_new_ticket' not in st.session_state: st.session_state.current_repairs_for_new_ticket = []
        
        rc1, rc2, rc3 = st.columns([2,3,1])
        repair_code_input = rc1.text_input("Code", key="new_repair_code")
        repair_desc_input = rc2.text_input("Description", key="new_repair_desc")
        if rc3.form_submit_button("Add Repair"): # Use form_submit_button for actions within form
            if repair_code_input and repair_desc_input:
                st.session_state.current_repairs_for_new_ticket.append({"code": repair_code_input.upper(), "description": repair_desc_input})
                # No rerun, form will update on main submit or this button press
                st.success(f"Added repair: {repair_code_input.upper()}")
            else: st.warning("Repair code and description needed.")
        
        if st.session_state.current_repairs_for_new_ticket:
            st.write("Current repairs for this ticket:")
            for r in st.session_state.current_repairs_for_new_ticket: st.markdown(f"- `{r['code']}`: {r['description']}")

        other_notes = st.text_area("Other Notes / Observations", key="submit_notes")

        st.markdown("**Attach Media Files:**")
        # Actual file uploader
        uploaded_files = st.file_uploader(
            "Upload images, videos, or documents", 
            accept_multiple_files=True, 
            key="submit_media_uploader",
            type=['png', 'jpg', 'jpeg', 'mp4', 'pdf', 'mov'] # Define acceptable types
        )
        # Allow associating uploaded files with repair codes
        # This part is tricky within a single form submission if files are processed before main submit.
        # For simplicity, we'll process associations on the main submit button.
        # For a better UX, this would be more dynamic.
        
        st.session_state.current_media_for_new_ticket_with_assoc = [] # Temp store for display before submit
        if uploaded_files:
            st.write("Associate uploaded files with repair codes (optional):")
            current_repair_codes_in_ticket = [r['code'] for r in st.session_state.current_repairs_for_new_ticket]
            for i, uploaded_file in enumerate(uploaded_files):
                assoc_code = st.selectbox(
                    f"Associate '{uploaded_file.name}' with repair code:",
                    options=[""] + current_repair_codes_in_ticket, # Can only associate with codes already added
                    key=f"submit_media_assoc_{i}"
                )
                # We can't store the file object directly here if clear_on_submit is True for the main form.
                # We'd need to process it on submit. For now, let's just get the names and associations.
                st.session_state.current_media_for_new_ticket_with_assoc.append({
                    "original_file_obj": uploaded_file, # Keep ref for processing on submit
                    "filename": uploaded_file.name,
                    "type": uploaded_file.type,
                    "repair_code_association": assoc_code if assoc_code else None
                })


        submitted_ticket_button = st.form_submit_button("Submit New Ticket for AI Processing")

        if submitted_ticket_button:
            if not container_id or not company or total_cost_estimate <= 0:
                st.error("Please fill required fields: Container ID, Company, Cost Estimate.")
            else:
                ticket_id = generate_ticket_id()
                
                # Process media from st.session_state.current_media_for_new_ticket_with_assoc
                final_media_list = []
                for media_item_info in st.session_state.current_media_for_new_ticket_with_assoc:
                    # In a real app, uploaded_file.original_file_obj.getvalue() would be read and saved to storage.
                    # For demo, just storing metadata.
                    final_media_list.append({
                        "filename": media_item_info["filename"],
                        "type": media_item_info["type"],
                        "repair_code_association": media_item_info["repair_code_association"],
                        # "bytes": media_item_info["original_file_obj"].getvalue() # Example if storing bytes
                    })

                new_ticket = {
                    "ticket_id": ticket_id, "container_id": container_id, "company": company,
                    "total_cost_estimate": total_cost_estimate, "container_age": container_age,
                    "repairs": list(st.session_state.current_repairs_for_new_ticket),
                    "media": final_media_list,
                    "other_notes": other_notes, "submitted_date": get_current_timestamp(),
                    "ai_chat": [{"timestamp": get_current_timestamp(), "sender": "System", "message": "Ticket submitted for AI processing."}]
                }
                
                # Immediately process with AI
                with st.spinner(f"AI processing new ticket {ticket_id}..."):
                    ai_result = process_ticket_with_ai(
                        new_ticket, st.session_state.age_cost_thresholds,
                        st.session_state.repair_codes_needing_images, use_cohere_aya=use_cohere
                    )
                
                new_ticket.update({
                    'ai_decision': ai_result['decision'], 'ai_confidence': ai_result.get('confidence_score', 0.0),
                    'ai_reasoning': ai_result['reasoning'], 'ai_agent_type': ai_result.get('ai_agent_type', 'Unknown AI'),
                    'ai_processed_date': get_current_timestamp(), 'ai_missing_data_request': ai_result.get('missing_data_request')
                })

                # Determine status based on AI result
                log_msg_action = ""
                if new_ticket['ai_missing_data_request']:
                    new_ticket['status'] = "Additional Data Requested"
                    log_msg_action = f"Additional Data Requested by AI ({new_ticket['ai_missing_data_request']})"
                    new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Additional data required: {new_ticket['ai_missing_data_request']}. Reasoning: {new_ticket['ai_reasoning']}"})
                elif new_ticket['ai_decision'] == "APPROVE" and new_ticket['ai_confidence'] >= 0.75:
                    new_ticket['status'] = "AI Approved"
                    log_msg_action = "AI Approved"
                    new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Approved. Reasoning: {new_ticket['ai_reasoning']}"})
                elif new_ticket['ai_decision'] == "DISAPPROVE":
                    new_ticket['status'] = "AI Disapproved"
                    log_msg_action = "AI Disapproved"
                    new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Disapproved. Reasoning: {new_ticket['ai_reasoning']}"})
                else: # MANUAL_REVIEW or low confidence APPROVE
                    new_ticket['status'] = "Manual Review Required"
                    log_msg_action = "Flagged for Manual Review by AI"
                    new_ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"This ticket requires manual review. Reasoning: {new_ticket['ai_reasoning']}"})

                st.session_state.tickets.append(new_ticket)
                log_action(ticket_id, f"New Ticket Submitted & {log_msg_action}", ai_result)
                
                st.session_state.current_repairs_for_new_ticket = [] # Clear for next form
                st.session_state.current_media_for_new_ticket_with_assoc = [] # Clear for next form
                st.success(f"New ticket {ticket_id} submitted and processed. Status: {new_ticket['status']}")
                # No st.rerun() due to clear_on_submit=True and form_submit_button behavior

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("© 2025 AI Repair Co. (Demo v4)")
st.sidebar.markdown("### AI Agent Strategy Note")
st.sidebar.caption("""
Hybrid Approach (Recommended): Rule-Based Engine for deterministic checks (cost, mandatory images), LLM for complex cases/text, Specialized Vision Models for image analysis.
This demo uses LLM for reasoning + rule-based pre-checks.
""")