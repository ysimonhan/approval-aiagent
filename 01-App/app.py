import streamlit as st
import pandas as pd
import datetime
import uuid
import cohere # For Cohere API
import json
import time # For simulating processing time

# --- Configuration ---
# (YC startups often use .env files or cloud secret managers for production)
COHERE_API_KEY = "9umxICMmVXpk6trBETGkfCPvHBzCV9TSjgyEVxWP"
COHERE_AYA_MODEL = "c4ai-aya-vision-32b" # Check Cohere documentation for latest vision-capable models on v2/chat

# Placeholder for custom model integration
CUSTOM_MODEL_CONFIG = {
    "api_key": "YOUR_CUSTOM_MODEL_API_KEY",
    "endpoint": "YOUR_CUSTOM_MODEL_ENDPOINT",
    # Add other necessary parameters for your custom model
}

# --- Helper Functions ---
def generate_ticket_id():
    return str(uuid.uuid4())[:8]

def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- AI Agent Logic ---

# Placeholder for your custom model integration
def call_custom_model(ticket_data, model_config):
    st.sidebar.info(f"Trying to call custom model with endpoint: {model_config.get('endpoint')}")
    # Simulate API call and response
    # In a real scenario, this would involve an HTTP request to your model's endpoint
    time.sleep(1) # Simulate network latency
    decision_options = ["APPROVE", "DISAPPROVE", "MANUAL_REVIEW"]
    decision = decision_options[hash(ticket_data['ticket_id']) % 3] # Pseudo-random decision
    confidence = 0.65 + (hash(ticket_data['ticket_id']) % 30) / 100.0 # Pseudo-random confidence

    reasoning = f"Custom model mock decision: Processed ticket {ticket_data['ticket_id']}. "
    if decision == "APPROVE":
        reasoning += "All checks passed according to custom model logic."
    elif decision == "DISAPPROVE":
        reasoning += "Disapproved due to custom model criteria (e.g., high cost for category)."
    else:
        reasoning += "Custom model suggests manual review due to borderline parameters."

    return {
        "decision": decision,
        "confidence_score": confidence,
        "reasoning": reasoning,
        "missing_images_for_codes": [] # Example
    }

def call_cohere_aya_model(ticket_data, age_cost_thresholds, repair_codes_needing_images):
    if not COHERE_API_KEY or COHERE_API_KEY == "YOUR_COHERE_API_KEY":
        st.error("Cohere API Key not configured. Please set it in the script.")
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.0,
            "reasoning": "Configuration error: Cohere API Key missing. Ticket requires manual review.",
            "missing_images_for_codes": []
        }

    co = cohere.Client(COHERE_API_KEY)

    # Construct the agent prompt
    prompt = f"""You are an AI Repair Ticket Approval Agent.
    Your task is to review the following container repair ticket and decide whether to 'APPROVE', 'DISAPPROVE', or flag for 'MANUAL_REVIEW'.
    Provide a confidence score (0.0 to 1.0) for your decision and a concise reasoning.

    Approval Criteria:
    1.  Cost vs. Container Age:
        {chr(10).join([f"    - {age_range}: Max approved cost ${threshold}" for age_range, threshold in age_cost_thresholds.items()])}
        If cost exceeds the limit for its age, tend towards disapproval or manual review.
    2.  Image Requirements:
        Certain repair codes require images as proof. These are: {', '.join(repair_codes_needing_images.keys())}.
        For each suggested repair:
            - If its code is in the list above AND an image for that specific repair is NOT provided, you MUST 'DISAPPROVE' the ticket and list the repair codes for which images are missing.

    Ticket Details:
    - Ticket ID: {ticket_data['ticket_id']}
    - Container ID: {ticket_data['container_id']}
    - Company: {ticket_data['company']}
    - Container Age (years): {ticket_data['container_age']}
    - Total Cost Estimate: ${ticket_data['total_cost_estimate']}
    - Suggested Repairs:"""

    repair_details_prompt = []
    missing_images_for_required_codes_internal_check = []

    for i, repair in enumerate(ticket_data.get('repairs', [])):
        code = repair.get('code', 'N/A')
        description = repair.get('description', 'No description')
        requires_image = code in repair_codes_needing_images
        image_provided_for_code = any(img_info.get('repair_code_association') == code for img_info in ticket_data.get('media', [])) if requires_image else True # Assume true if not required

        repair_details_prompt.append(
            f"  - Repair Code: {code}, Description: {description} (Requires Image: {requires_image}, Image Provided for this code: {image_provided_for_code})"
        )
        if requires_image and not image_provided_for_code:
            missing_images_for_required_codes_internal_check.append(code)

    prompt += "\n" + "\n".join(repair_details_prompt)
    prompt += f"\n- Other Notes: {ticket_data.get('other_notes', 'None')}"
    prompt += f"\n- Media Provided: {len(ticket_data.get('media', []))} files. (Assume specific images for repairs are checked as per 'Image Provided for this code' flags above)."

    # Pre-emptive check for missing images to guide the LLM or even short-circuit
    if missing_images_for_required_codes_internal_check:
        return {
            "decision": "DISAPPROVE",
            "confidence_score": 1.0, # High confidence due to rule-based check
            "reasoning": f"Disapproved: Mandatory images missing for repair codes: {', '.join(missing_images_for_required_codes_internal_check)}.",
            "missing_images_for_codes": missing_images_for_required_codes_internal_check
        }

    prompt += """

Based on all the above, provide your response strictly in the following JSON format:
{
  "decision": "APPROVE" / "DISAPPROVE" / "MANUAL_REVIEW",
  "confidence_score": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation for the decision, including specific criteria met or failed>",
  "missing_images_for_codes": ["<repair_code_1>", "<repair_code_2>"] // Empty list if no images are missing or if decision is not DISAPPROVE due to missing images.
}
"""
    try:
        # For Aya Vision, if you were to send image data/URLs, it would be part of the 'message' structure,
        # often in a 'documents' field or similar, depending on Cohere's exact API spec for multimodal.
        # Here, we are relying on the textual description of image presence.
        st.sidebar.info(f"Calling Cohere Aya ({COHERE_AYA_MODEL})...")
        response = co.chat(
            model=COHERE_AYA_MODEL,
            message=prompt,
            # If you had image URLs or base64 data and the API supports it:
            # documents=[{'title': 'repair_image_1.jpg', 'snippet': 'base64_encoded_string_or_url'}]
            # The current Aya models and chat endpoint might have specific ways to handle images.
            # Refer to latest Cohere documentation for multimodal message structuring.
            # For this example, we've embedded image presence info directly in the text prompt.
        )
        st.sidebar.success("Cohere Aya response received.")

        # Attempt to parse the JSON from the response text
        # The actual response object structure from co.chat is response.text for the message content
        ai_response_text = response.text
        # LLMs can sometimes add extra text around the JSON, so try to extract it
        json_start = ai_response_text.find('{')
        json_end = ai_response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            parsed_response = json.loads(ai_response_text[json_start:json_end])
            # Basic validation of the parsed structure
            if not all(key in parsed_response for key in ["decision", "confidence_score", "reasoning"]):
                raise ValueError("LLM response missing required JSON keys.")
            return parsed_response
        else:
            st.error(f"Could not parse JSON from Cohere response: {ai_response_text}")
            return {
                "decision": "MANUAL_REVIEW",
                "confidence_score": 0.1,
                "reasoning": f"Error: Could not parse AI response. Raw response: {ai_response_text}",
                "missing_images_for_codes": []
            }

    except cohere.CohereError as e:
        st.error(f"Cohere API Error: {e}")
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.0,
            "reasoning": f"Cohere API error: {e}. Ticket requires manual review.",
            "missing_images_for_codes": []
        }
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error from Cohere response: {e}. Response: {ai_response_text}")
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.1,
            "reasoning": f"Error: AI response was not valid JSON. Raw response: {ai_response_text}",
            "missing_images_for_codes": []
        }
    except Exception as e:
        st.error(f"An unexpected error occurred while calling Cohere API: {e}")
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.0,
            "reasoning": f"Unexpected error: {e}. Ticket requires manual review.",
            "missing_images_for_codes": []
        }

def process_ticket_with_ai(ticket_data, age_cost_thresholds, repair_codes_needing_images, use_cohere_aya=True):
    if use_cohere_aya:
        # Check for image requirements before calling LLM if it's a strict rule
        missing_required_images = []
        for repair in ticket_data.get('repairs', []):
            code = repair.get('code')
            if code in repair_codes_needing_images:
                # Check if an image is associated with this specific repair code
                image_provided_for_this_code = any(
                    media_item.get('repair_code_association') == code
                    for media_item in ticket_data.get('media', [])
                )
                if not image_provided_for_this_code:
                    missing_required_images.append(code)
        
        if missing_required_images:
            return {
                "decision": "DISAPPROVE",
                "confidence_score": 1.0, # High confidence due to rule
                "reasoning": f"Rule-based disapproval: Mandatory images missing for repair codes: {', '.join(missing_required_images)}.",
                "missing_images_for_codes": missing_required_images,
                "ai_agent_type": "Rule-Based Pre-check"
            }
        
        ai_result = call_cohere_aya_model(ticket_data, age_cost_thresholds, repair_codes_needing_images)
        ai_result["ai_agent_type"] = f"Cohere Aya ({COHERE_AYA_MODEL})"
    else:
        ai_result = call_custom_model(ticket_data, CUSTOM_MODEL_CONFIG) # Using the placeholder
        ai_result["ai_agent_type"] = "Custom Model (Placeholder)"
    return ai_result

# --- Streamlit App Initialization ---
st.set_page_config(layout="wide", page_title="AI Repair Ticket Approval")

# --- Initialize Session State ---
if 'tickets' not in st.session_state:
    st.session_state.tickets = [] # All tickets
if 'logs' not in st.session_state:
    st.session_state.logs = [] # Audit trail
if 'feedback' not in st.session_state:
    st.session_state.feedback = [] # For training tab

# Default editable thresholds (YC startups value configuration)
if 'age_cost_thresholds' not in st.session_state:
    st.session_state.age_cost_thresholds = {
        "New (0-2 years)": 1000,
        "Medium (3-5 years)": 700,
        "Old (6-8 years)": 400,
        "Very Old (>8 years)": 200,
    }
if 'repair_codes_needing_images' not in st.session_state:
    st.session_state.repair_codes_needing_images = {
        "DMG001": "Severe Dent Repair",
        "CRK003": "Frame Crack Assessment",
        # Add more codes and their descriptions that require images
    }

# Sample Data (To make it easy to add new columns, new tickets are dictionaries)
def add_sample_tickets():
    if not st.session_state.tickets: # Add only if empty
        sample_tickets_data = [
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON001", "company": "Maersk", "total_cost_estimate": 800,
                "container_age": 1, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "DNT001", "description": "Minor dent on side panel"}, {"code": "SCT002", "description": "Scratch on door"}],
                "media": [{"filename": "img1.jpg", "type": "image", "repair_code_association": "DNT001"}, {"filename": "vid1.mp4", "type": "video"}], # Example of associating media
                "ai_chat": [], "other_notes": "Standard wear and tear."
            },
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON002", "company": "MSC", "total_cost_estimate": 1200,
                "container_age": 4, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "DMG001", "description": "Significant structural damage"}, {"code": "FLR001", "description": "Floorboard replacement"}],
                "media": [], # DMG001 requires an image, but it's missing
                "ai_chat": [], "other_notes": "Needs urgent assessment."
            },
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON003", "company": "Cosco", "total_cost_estimate": 300,
                "container_age": 7, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "RST005", "description": "Surface rust treatment"}],
                "media": [{"filename": "rust_overview.jpg", "type": "image"}],
                "ai_chat": [], "other_notes": ""
            },
            {
                "ticket_id": generate_ticket_id(), "container_id": "CON004", "company": "CGM", "total_cost_estimate": 900,
                "container_age": 2, "status": "Pending AI Review", "submitted_date": get_current_timestamp(),
                "repairs": [{"code": "DMG001", "description": "Frame damage report"}, {"code": "WEL001", "description": "Welding required"}],
                "media": [{"filename": "frame_damage_con004.jpg", "type": "image", "repair_code_association": "DMG001"}],
                "ai_chat": [], "other_notes": "Impact damage noted."
            }
        ]
        st.session_state.tickets.extend(sample_tickets_data)

add_sample_tickets() # Load sample data

# --- Sidebar ---
st.sidebar.title("🚢 Repair Approval Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Home", "Approvals", "AI Training & Settings", "Submit New Ticket"])

st.sidebar.markdown("---")
st.sidebar.subheader("AI Agent Settings")
use_cohere = st.sidebar.checkbox("Use Cohere Aya Model", value=True)
if not use_cohere:
    st.sidebar.caption("Currently using Custom Model Placeholder.")

# --- Page Implementations ---

def display_ticket_details(ticket, expand_details=False, show_actions=True):
    with st.expander(f"Ticket {ticket['ticket_id']} ({ticket['company']} - {ticket['container_id']}) - Status: {ticket['status']}", expanded=expand_details):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company:** {ticket['company']}")
            st.write(f"**Container ID:** {ticket['container_id']}")
            st.write(f"**Container Age:** {ticket['container_age']} years")
            st.write(f"**Total Cost Estimate:** ${ticket['total_cost_estimate']:.2f}")
            st.write(f"**Submitted:** {ticket.get('submitted_date', 'N/A')}")
            st.write(f"**Last AI Update:** {ticket.get('ai_processed_date', 'N/A')}")

        with col2:
            st.write("**Suggested Repairs:**")
            if ticket.get('repairs'):
                for repair in ticket['repairs']:
                    st.markdown(f"- `{repair.get('code', 'N/A')}`: {repair.get('description', 'No description')}")
            else:
                st.write("No repairs listed.")

            st.write("**Media Files:**")
            if ticket.get('media'):
                for media_item in ticket['media']:
                    assoc_code = f" (for {media_item.get('repair_code_association')})" if media_item.get('repair_code_association') else ""
                    st.markdown(f"- {media_item.get('filename', 'N/A')} ({media_item.get('type', 'N/A')}){assoc_code}")
            else:
                st.write("No media files attached.")
        
        st.markdown(f"**Other Notes:** {ticket.get('other_notes', 'None')}")

        if ticket.get('ai_reasoning'):
            st.info(f"**AI Agent ({ticket.get('ai_agent_type', 'N/A')} on {ticket.get('ai_processed_date', '')}):**\n"
                    f"Decision: **{ticket.get('ai_decision', 'N/A')}** (Confidence: {ticket.get('ai_confidence', 0.0):.2f})\n"
                    f"Reasoning: {ticket.get('ai_reasoning', 'N/A')}")
            if ticket.get('ai_missing_images'):
                st.warning(f"AI noted missing images for codes: {', '.join(ticket['ai_missing_images'])}")

        st.write("**Communication Log / AI Chat:**")
        if not ticket.get('ai_chat'):
            st.caption("No messages yet.")
        for chat_msg in ticket.get('ai_chat', []):
            st.text_area("", value=f"[{chat_msg['timestamp']}] {chat_msg['sender']}: {chat_msg['message']}", height=50, disabled=True, key=f"chat_{ticket['ticket_id']}_{chat_msg['timestamp']}")

        if show_actions and ticket['status'] not in ["AI Approved", "AI Disapproved", "Manually Approved", "Manually Disapproved"]:
             # Only show AI process button if not yet processed by AI or needs reprocessing
            if ticket['status'] == "Pending AI Review" or ticket['status'] == "Awaiting Images":
                if st.button(f"Process with AI Now", key=f"process_{ticket['ticket_id']}"):
                    with st.spinner(f"AI processing ticket {ticket['ticket_id']}..."):
                        ai_result = process_ticket_with_ai(
                            ticket,
                            st.session_state.age_cost_thresholds,
                            st.session_state.repair_codes_needing_images,
                            use_cohere_aya=use_cohere
                        )

                    ticket['ai_decision'] = ai_result['decision']
                    ticket['ai_confidence'] = ai_result.get('confidence_score', 0.0)
                    ticket['ai_reasoning'] = ai_result['reasoning']
                    ticket['ai_agent_type'] = ai_result.get('ai_agent_type', 'Unknown AI')
                    ticket['ai_processed_date'] = get_current_timestamp()
                    ticket['ai_missing_images'] = ai_result.get('missing_images_for_codes', [])

                    if ai_result['decision'] == "APPROVE" and ticket['ai_confidence'] >= 0.75:
                        ticket['status'] = "AI Approved"
                        log_message = f"Ticket {ticket['ticket_id']} AI Approved."
                        ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Approved. Reasoning: {ticket['ai_reasoning']}"})
                    elif ai_result['decision'] == "DISAPPROVE":
                        ticket['status'] = "AI Disapproved"
                        log_message = f"Ticket {ticket['ticket_id']} AI Disapproved."
                        if ticket['ai_missing_images']:
                             ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Disapproved. Mandatory images missing for repair codes: {', '.join(ticket['ai_missing_images'])}. Please upload them and resubmit."})
                             ticket['status'] = "Awaiting Images" # Special status
                        else:
                            ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Disapproved. Reasoning: {ticket['ai_reasoning']}"})
                    else: # MANUAL_REVIEW or low confidence
                        ticket['status'] = "Manual Review Required"
                        log_message = f"Ticket {ticket['ticket_id']} flagged for Manual Review by AI."
                        ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"This ticket requires manual review. Reasoning: {ticket['ai_reasoning']}"})

                    st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": ticket['ticket_id'], "action": log_message, "details": ai_result})
                    st.rerun()
        
        # Manual Override Actions (visible for AI processed or Manual Review tickets)
        if ticket['status'] in ["AI Approved", "AI Disapproved", "Manual Review Required", "Awaiting Images"]:
            st.markdown("---")
            st.write("**Manual Actions:**")
            b_col1, b_col2, b_col3 = st.columns(3)
            if b_col1.button("Manually Approve", key=f"manual_approve_{ticket['ticket_id']}"):
                ticket['status'] = "Manually Approved"
                ticket['manual_override_by'] = "User" # Add user info in a real app
                ticket['manual_override_date'] = get_current_timestamp()
                ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "System", "message": "Ticket Manually Approved by User."})
                st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": ticket['ticket_id'], "action": "Manually Approved", "user": "User"})
                st.rerun()
            if b_col2.button("Manually Disapprove", key=f"manual_disapprove_{ticket['ticket_id']}"):
                ticket['status'] = "Manually Disapproved"
                ticket['manual_override_by'] = "User"
                ticket['manual_override_date'] = get_current_timestamp()
                ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "System", "message": "Ticket Manually Disapproved by User."})
                st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": ticket['ticket_id'], "action": "Manually Disapproved", "user": "User"})
                st.rerun()
            if ticket['status'] in ["AI Approved", "AI Disapproved"]:
                 if b_col3.button("Revert to Manual Review", key=f"revert_{ticket['ticket_id']}"):
                    ticket['original_ai_status'] = ticket['status']
                    ticket['status'] = "Manual Review Required"
                    ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "System", "message": f"AI decision ({ticket['original_ai_status']}) reverted by User. Ticket moved to Manual Review."})
                    st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": ticket['ticket_id'], "action": "AI Decision Reverted to Manual Review", "user": "User"})
                    st.rerun()
        
        # For tickets awaiting images, allow attaching "new" images (simulation)
        if ticket['status'] == "Awaiting Images":
            st.markdown("---")
            st.write("**Upload Missing Images (Simulated):**")
            # In a real app, this would be st.file_uploader
            uploaded_for_code = st.multiselect("Mark images as uploaded for repair codes:", ticket.get('ai_missing_images', []), key=f"upload_sim_{ticket['ticket_id']}")
            if st.button("Simulate Image Upload & Resubmit", key=f"resubmit_img_{ticket['ticket_id']}"):
                if uploaded_for_code:
                    for code in uploaded_for_code:
                        # Simulate adding media
                        ticket.setdefault('media', []).append({
                            "filename": f"new_image_for_{code}.jpg", 
                            "type": "image", 
                            "repair_code_association": code,
                            "uploaded_timestamp": get_current_timestamp()
                        })
                    ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "System", "message": f"Simulated images uploaded for codes: {', '.join(uploaded_for_code)}. Resubmitting for AI review."})
                    ticket['status'] = "Pending AI Review" # Set back to pending for AI
                    # Clear previous AI decision that was due to missing images to allow re-evaluation
                    ticket['ai_decision'] = None
                    ticket['ai_reasoning'] = None
                    ticket['ai_missing_images'] = []
                    st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": ticket['ticket_id'], "action": f"Simulated images uploaded for {', '.join(uploaded_for_code)}. Resubmitted.", "user": "User"})
                    st.rerun()
                else:
                    st.warning("Please select codes for which images are 'uploaded'.")


if page == "Home":
    st.header("Dashboard Overview")
    
    pending_ai_review = [t for t in st.session_state.tickets if t['status'] == "Pending AI Review"]
    manual_review_needed = [t for t in st.session_state.tickets if t['status'] == "Manual Review Required"]
    awaiting_images = [t for t in st.session_state.tickets if t['status'] == "Awaiting Images"]
    
    ai_approved_count = len([t for t in st.session_state.tickets if t['status'] == "AI Approved"])
    ai_disapproved_count = len([t for t in st.session_state.tickets if t['status'] == "AI Disapproved"])
    manual_approved_count = len([t for t in st.session_state.tickets if t['status'] == "Manually Approved"])
    manual_disapproved_count = len([t for t in st.session_state.tickets if t['status'] == "Manually Disapproved"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets Pending AI Processing", len(pending_ai_review))
    col2.metric("Tickets for Manual Review", len(manual_review_needed))
    col3.metric("Tickets Awaiting Images", len(awaiting_images))

    st.subheader("Completion Status")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("AI Approved", ai_approved_count)
    col_b.metric("AI Disapproved", ai_disapproved_count)
    col_c.metric("Manually Approved", manual_approved_count)
    col_d.metric("Manually Disapproved", manual_disapproved_count)

    total_to_do = len(pending_ai_review) + len(manual_review_needed) + len(awaiting_images)
    # Simple time estimation (YC startups often do data-driven estimations)
    avg_ai_time_per_ticket = 5 # seconds (simulated)
    avg_manual_time_per_ticket = 180 # seconds
    estimated_time_seconds = (len(pending_ai_review) * avg_ai_time_per_ticket) + \
                             ((len(manual_review_needed) + len(awaiting_images)) * avg_manual_time_per_ticket)
    
    st.info(f"**Total tickets remaining:** {total_to_do}")
    if total_to_do > 0:
        st.info(f"**Estimated time to clear queue:** {datetime.timedelta(seconds=estimated_time_seconds)}")
    else:
        st.success("All tickets processed!")

    st.subheader("Recent Activity Log (Last 10)")
    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs).sort_values(by="timestamp", ascending=False)
        st.dataframe(log_df.head(10), use_container_width=True)
    else:
        st.write("No activity logged yet.")


elif page == "Approvals":
    st.header("Repair Ticket Approval Queues")

    tab_titles = ["Pending AI Review", "Manual Review Required", "AI Approved", "AI Disapproved", "Manually Processed", "Awaiting Images"]
    tabs = st.tabs(tab_titles)

    ticket_categories = {
        "Pending AI Review": [t for t in st.session_state.tickets if t['status'] == "Pending AI Review"],
        "Manual Review Required": [t for t in st.session_state.tickets if t['status'] == "Manual Review Required"],
        "AI Approved": [t for t in st.session_state.tickets if t['status'] == "AI Approved"],
        "AI Disapproved": [t for t in st.session_state.tickets if t['status'] == "AI Disapproved"],
        "Manually Processed": [t for t in st.session_state.tickets if t['status'] in ["Manually Approved", "Manually Disapproved"]],
        "Awaiting Images": [t for t in st.session_state.tickets if t['status'] == "Awaiting Images"]
    }

    for i, title in enumerate(tab_titles):
        with tabs[i]:
            st.subheader(f"{title} ({len(ticket_categories[title])})")
            if not ticket_categories[title]:
                st.info(f"No tickets in '{title}' queue.")
            else:
                # Batch AI Processing for "Pending AI Review"
                if title == "Process Pending AI":
                    num_to_process = st.number_input("Number of pending tickets to process with AI in batch:", min_value=1, max_value=len(ticket_categories[title]), value=min(5, len(ticket_categories[title])), step=1, key="batch_process_num")
                    if st.button("Process Batch with AI", key="batch_process_ai"):
                        processed_count = 0
                        with st.spinner(f"Processing batch of {num_to_process} tickets..."):
                            for ticket in ticket_categories[title][:num_to_process]:
                                if ticket['status'] == "Pending AI Review": # Double check status
                                    ai_result = process_ticket_with_ai(
                                        ticket,
                                        st.session_state.age_cost_thresholds,
                                        st.session_state.repair_codes_needing_images,
                                        use_cohere_aya=use_cohere
                                    )
                                    ticket['ai_decision'] = ai_result['decision']
                                    ticket['ai_confidence'] = ai_result.get('confidence_score', 0.0)
                                    ticket['ai_reasoning'] = ai_result['reasoning']
                                    ticket['ai_agent_type'] = ai_result.get('ai_agent_type', 'Unknown AI')
                                    ticket['ai_processed_date'] = get_current_timestamp()
                                    ticket['ai_missing_images'] = ai_result.get('missing_images_for_codes', [])

                                    if ai_result['decision'] == "APPROVE" and ticket['ai_confidence'] >= 0.75:
                                        ticket['status'] = "AI Approved"
                                        log_message = f"Ticket {ticket['ticket_id']} AI Approved (Batch)."
                                        ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Approved. Reasoning: {ticket['ai_reasoning']}"})
                                    elif ai_result['decision'] == "DISAPPROVE":
                                        ticket['status'] = "AI Disapproved"
                                        log_message = f"Ticket {ticket['ticket_id']} AI Disapproved (Batch)."
                                        if ticket['ai_missing_images']:
                                            ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Disapproved. Mandatory images missing for repair codes: {', '.join(ticket['ai_missing_images'])}. Please upload them and resubmit."})
                                            ticket['status'] = "Awaiting Images"
                                        else:
                                          ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"Ticket Disapproved. Reasoning: {ticket['ai_reasoning']}"})
                                    else:
                                        ticket['status'] = "Manual Review Required"
                                        log_message = f"Ticket {ticket['ticket_id']} flagged for Manual Review by AI (Batch)."
                                        ticket['ai_chat'].append({"timestamp": get_current_timestamp(), "sender": "AI Agent", "message": f"This ticket requires manual review. Reasoning: {ticket['ai_reasoning']}"})
                                    
                                    st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": ticket['ticket_id'], "action": log_message, "details": ai_result})
                                    processed_count +=1
                        st.success(f"Batch processing complete. {processed_count} tickets processed.")
                        st.rerun()
                
                # Display individual tickets
                for ticket in sorted(ticket_categories[title], key=lambda x: x.get('submitted_date', ''), reverse=True):
                    display_ticket_details(ticket, show_actions=True)
                    st.markdown("---")


elif page == "AI Training & Settings":
    st.header("AI Evaluation & Configuration")

    st.subheader("AI Decision Thresholds")
    st.markdown("Configure the cost approval limits based on container age. The AI will use these.")
    
    # Editable thresholds
    updated_thresholds = {}
    for age_range, current_threshold in st.session_state.age_cost_thresholds.items():
        updated_thresholds[age_range] = st.number_input(
            f"Max Cost for {age_range}", 
            min_value=0, 
            value=current_threshold, 
            step=50,
            key=f"thresh_{age_range.replace(' ', '_')}"
        )
    if st.button("Save Cost Thresholds"):
        st.session_state.age_cost_thresholds = updated_thresholds
        st.success("Cost thresholds updated!")
        st.session_state.logs.append({"timestamp": get_current_timestamp(), "action": "AI Cost Thresholds Updated", "details": updated_thresholds})
        st.rerun()

    st.markdown("---")
    st.subheader("Repair Codes Requiring Images")
    st.markdown("Define which repair codes mandatorily require an image for approval.")
    
    current_img_req_codes = st.session_state.repair_codes_needing_images
    # Display current codes
    if current_img_req_codes:
        st.write("Current codes requiring images:")
        for code, desc in current_img_req_codes.items():
            st.markdown(f"- `{code}`: {desc}")
    else:
        st.write("No repair codes currently configured to require images.")

    with st.form("add_image_req_code_form"):
        new_code = st.text_input("New Repair Code (e.g., DMG002)")
        new_code_desc = st.text_input("Description for new code (e.g., Critical Weld Point)")
        submitted_new_code = st.form_submit_button("Add Image Requirement")
        if submitted_new_code and new_code and new_code_desc:
            st.session_state.repair_codes_needing_images[new_code.upper()] = new_code_desc
            st.success(f"Added '{new_code.upper()}' to image requirement list.")
            st.session_state.logs.append({"timestamp": get_current_timestamp(), "action": "AI Image Requirement Added", "details": {new_code.upper(): new_code_desc}})
            st.rerun()
        elif submitted_new_code:
            st.error("Both code and description are required.")
    
    # Option to remove a code
    if current_img_req_codes:
        code_to_remove = st.selectbox("Select code to remove from image requirement list:", options=[""] + list(current_img_req_codes.keys()))
        if st.button("Remove Selected Code Requirement") and code_to_remove:
            del st.session_state.repair_codes_needing_images[code_to_remove]
            st.success(f"Removed '{code_to_remove}' from image requirement list.")
            st.session_state.logs.append({"timestamp": get_current_timestamp(), "action": "AI Image Requirement Removed", "details": {"code_removed": code_to_remove}})
            st.rerun()


    st.markdown("---")
    st.subheader("Evaluate AI Decisions (for Fine-tuning Data Collection)")
    st.markdown("Review AI-processed tickets and provide feedback. This data can be used to fine-tune the AI model in the future.")

    ai_processed_tickets = [t for t in st.session_state.tickets if t['status'] in ["AI Approved", "AI Disapproved"] and 'ai_decision' in t]
    
    if not ai_processed_tickets:
        st.info("No AI-processed tickets available for evaluation yet.")
    else:
        for ticket in ai_processed_tickets:
            with st.container(border=True): # Visually group each ticket
                st.write(f"**Ticket ID: {ticket['ticket_id']}** (AI Decision: {ticket['ai_decision']} with {ticket['ai_confidence']:.2f} confidence)")
                st.caption(f"AI Reasoning: {ticket['ai_reasoning']}")
                
                feedback_key_suffix = f"feedback_{ticket['ticket_id']}"
                # Check if feedback already given for this ticket
                existing_feedback = next((f for f in st.session_state.feedback if f['ticket_id'] == ticket['ticket_id']), None)

                if existing_feedback:
                    st.success(f"Feedback submitted: {existing_feedback['evaluation']} ({existing_feedback.get('comment', '')})")
                else:
                    cols = st.columns([1,1,3])
                    thumbs_up = cols[0].button("👍 Good Decision", key=f"up_{feedback_key_suffix}")
                    thumbs_down = cols[1].button("👎 Bad Decision", key=f"down_{feedback_key_suffix}")
                    comment = cols[2].text_input("Optional comment", key=f"comment_{feedback_key_suffix}")

                    if thumbs_up or thumbs_down:
                        evaluation = "Good" if thumbs_up else "Bad"
                        feedback_data = {
                            "ticket_id": ticket['ticket_id'],
                            "ai_decision": ticket['ai_decision'],
                            "ai_confidence": ticket['ai_confidence'],
                            "ai_reasoning": ticket['ai_reasoning'],
                            "evaluation": evaluation,
                            "comment": comment,
                            "timestamp": get_current_timestamp()
                        }
                        st.session_state.feedback.append(feedback_data)
                        st.session_state.logs.append({"timestamp": get_current_timestamp(), "action": "AI Feedback Submitted", "details": feedback_data})
                        st.success(f"Feedback '{evaluation}' recorded for ticket {ticket['ticket_id']}.")
                        st.rerun() # Rerun to show feedback submitted message
                st.markdown("---")

    if st.session_state.feedback:
        st.subheader("Collected Feedback Data")
        feedback_df = pd.DataFrame(st.session_state.feedback)
        st.dataframe(feedback_df, use_container_width=True)
        # Provide download for YC-style data utility
        st.download_button(
            label="Download Feedback Data (CSV)",
            data=feedback_df.to_csv(index=False).encode('utf-8'),
            file_name='ai_feedback_data.csv',
            mime='text/csv',
        )

elif page == "Submit New Ticket":
    st.header("Submit New Repair Ticket ")
    st.markdown("Fill in the details for the new repair ticket. Fields marked with * are notionally required.")

    with st.form("new_ticket_form", clear_on_submit=True):
        # Easily extensible columns: Just add more st.text_input, st.number_input etc. here
        c1, c2 = st.columns(2)
        container_id = c1.text_input("Container ID*")
        company = c2.selectbox("Shipping Company*", ["Maersk", "Cosco", "CGM", "MSC", "Hapag", "Other"])
        if company == "Other":
            company = c2.text_input("Specify Other Company")

        total_cost_estimate = c1.number_input("Total Cost Estimate ($)*", min_value=0.0, step=10.0, format="%.2f")
        container_age = c2.number_input("Container Age (years)*", min_value=0, max_value=50, step=1)
        
        st.markdown("**Suggested Repairs (add one by one):**")
        # Simple way to add multiple repairs
        if 'current_repairs_for_new_ticket' not in st.session_state:
            st.session_state.current_repairs_for_new_ticket = []

        rc1, rc2 = st.columns([2, 3])
        repair_code_input = rc1.text_input("Repair Code (e.g., DNT001)", key="new_repair_code")
        repair_desc_input = rc2.text_input("Repair Description", key="new_repair_desc")
        
        # if rc3.button("Add Repair", key="add_repair_item_button", use_container_width=True):
        #     if repair_code_input and repair_desc_input:
        #         st.session_state.current_repairs_for_new_ticket.append({"code": repair_code_input, "description": repair_desc_input})
        #         # No rerun here, form will handle it
        #     else:
        #         st.warning("Repair code and description needed to add.")
        
        # Hidden submit button for adding repairs
        add_repair_clicked = st.form_submit_button("Add Repair")
        if add_repair_clicked:
            if repair_code_input and repair_desc_input:
                st.session_state.current_repairs_for_new_ticket.append({"code": repair_code_input, "description": repair_desc_input})
                st.success(f"Added repair: `{repair_code_input}` - {repair_desc_input}")
            else:
                st.warning("Repair code and description are required to add a repair.")
        
        # Display current repairs added to the ticket
        if st.session_state.current_repairs_for_new_ticket:
            st.write("Current repairs added to ticket:")
            for i, r in enumerate(st.session_state.current_repairs_for_new_ticket):
                st.markdown(f"- `{r['code']}`: {r['description']}")

        # Other fields
        other_notes = st.text_area("Other Notes / Observations")

        # Simplified media upload - in reality, use st.file_uploader(accept_multiple_files=True)
        # This simulation just notes filenames and which repair code they are for (if any)
        st.markdown("**Media Files (Simulated Attachment):**")
        if 'current_media_for_new_ticket' not in st.session_state:
            st.session_state.current_media_for_new_ticket = []
        
        mc1, mc2, mc3 = st.columns([2,2,2])
        media_filename_input = mc1.text_input("Filename (e.g., image.jpg, video.mp4)", key="new_media_file")
        media_type_input = mc2.selectbox("Type", ["image", "video", "document"], key="new_media_type")
        
        # Allow associating media with a specific repair code from the ones added above
        current_repair_codes_in_ticket = [r['code'] for r in st.session_state.current_repairs_for_new_ticket]
        media_repair_assoc_input = mc3.selectbox("Associate with Repair Code (Optional)", options=[""] + current_repair_codes_in_ticket, key="new_media_assoc")

        # if mc4.button("Add Media Item", key="add_media_item_button", use_container_width=True):
        #     if media_filename_input:
        #         st.session_state.current_media_for_new_ticket.append({
        #             "filename": media_filename_input, 
        #             "type": media_type_input,
        #             "repair_code_association": media_repair_assoc_input if media_repair_assoc_input else None
        #         })
        #     else:
        #         st.warning("Filename is required to add media.")

        # Hidden submit button for adding media
        add_media_clicked = st.form_submit_button("Add Media Item")
        if add_media_clicked:
            if media_filename_input:
                st.session_state.current_media_for_new_ticket.append({
                    "filename": media_filename_input, 
                    "type": media_type_input,
                    "repair_code_association": media_repair_assoc_input if media_repair_assoc_input else None
                })
                st.success(f"Added media: `{media_filename_input}` ({media_type_input})")
            else:
                st.warning("Filename is required to add media.")
        
        # Display current media added to the ticket
        if st.session_state.current_media_for_new_ticket:
            st.write("Current media added to ticket:")
            for i, m in enumerate(st.session_state.current_media_for_new_ticket):
                assoc_text = f" (for {m['repair_code_association']})" if m['repair_code_association'] else ""
                st.markdown(f"- {m['filename']} ({m['type']}){assoc_text}")


        submitted = st.form_submit_button("Submit New Ticket")

        if submitted:
            if not container_id or not company or total_cost_estimate <= 0:
                st.error("Please fill in all required fields: Container ID, Company, and a valid Total Cost Estimate.")
            else:
                new_ticket = {
                    "ticket_id": generate_ticket_id(),
                    "container_id": container_id,
                    "company": company,
                    "total_cost_estimate": total_cost_estimate,
                    "container_age": container_age,
                    "repairs": list(st.session_state.current_repairs_for_new_ticket), # Take a copy
                    "media": list(st.session_state.current_media_for_new_ticket), # Take a copy
                    "other_notes": other_notes,
                    "status": "Pending AI Review", # Initial status
                    "submitted_date": get_current_timestamp(),
                    "ai_chat": [{"timestamp": get_current_timestamp(), "sender": "System", "message": "Ticket submitted."}]
                }
                st.session_state.tickets.append(new_ticket)
                st.session_state.logs.append({"timestamp": get_current_timestamp(), "ticket_id": new_ticket['ticket_id'], "action": "New Ticket Submitted", "details": {"company": company, "cost": total_cost_estimate}})
                
                # Clear form-specific session state
                st.session_state.current_repairs_for_new_ticket = []
                st.session_state.current_media_for_new_ticket = []
                
                st.success(f"New repair ticket {new_ticket['ticket_id']} submitted successfully!")
                # No st.rerun() here, clear_on_submit and form behavior handles refresh of form elements.

# --- Footer (YC style might be minimal or link to terms/status) ---
st.sidebar.markdown("---")
st.sidebar.info("© 2025 AI Repair Co. (Demo)")

# --- Alternative AI Agent Suggestion (as requested) ---
# (This would be part of your documentation or a separate discussion)
st.sidebar.markdown("### AI Agent Strategy Note")
st.sidebar.caption("""
While an LLM offers great flexibility for reasoning and understanding unstructured data, consider these alternatives or complements:

1.  **Hybrid Approach (Recommended for Robustness):**
    * **Rule-Based Engine:** For deterministic criteria (e.g., cost thresholds, mandatory image checks based on codes). This is fast, reliable, and auditable for simple rules.
    * **LLM for Complex Cases:** Use the LLM for nuanced decisions, interpreting free-text notes, generating human-like explanations, or handling situations not covered by simple rules. The LLM could also be used to *suggest* rules or identify patterns.
    * **Specialized Vision Models:** For actual image content analysis (e.g., assessing damage severity from photos), a dedicated computer vision model (trained for specific damage types) would be more accurate than a general multimodal LLM for detailed visual tasks. The output of this vision model (e.g., "severe corrosion detected") can then be fed into the rule engine or LLM.

2.  **Structured Data + Simpler ML Models:**
    If ticket data can be highly structured and features well-engineered (e.g., numerical damage scores, categorical repair types), traditional ML models (like Gradient Boosting, Random Forests, or even logistic regression) could be trained for approval/disapproval classification. These are often more transparent and computationally cheaper than large LLMs but less flexible with unstructured text or novel scenarios.

This demo primarily uses an LLM for end-to-end reasoning but includes rule-based pre-checks for image requirements. A production system would benefit from a more layered approach.
""")