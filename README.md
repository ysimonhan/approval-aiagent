# 🚢 AI Repair Ticket Approval Platform

A simple and straightforward Streamlit app that is powered by a mix of LLM, OCR, and object detection to empower approval teams to manage the overwhelming amount of repair tickets of shipment containers :)

## ⚡ Key Features

- **AI-Driven Decision Engine** Combines rule-based cost vs. age checks with Cohere Aya for nuanced reasoning and YOLOv8 object detection to analyze damage images in real time.  
- **Interactive Dashboard** Track and manage tickets through “Manual Review,” “Additional Data Requested,” “AI Approved,” or “AI Disapproved” queues—all with one click.  
- **Customizable Settings** Define cost thresholds by container age and specify which repair codes require image evidence. Update on the fly via the AI Training & Settings panel.  
- **Seamless SOP Integration** Upload PDFs of your Standard Operating Procedures to automatically extract repair codes needing visual documentation using Mistral OCR.  
- **In-App Chat & Collaboration** Chat with the AI agent for clarifications or loop in your team with manual messages—contextual history keeps everyone on the same page.  
