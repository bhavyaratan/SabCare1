import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from db import SessionLocal, Patient
from twilio_call import twilio_call_service, make_call_and_play_script
from medgemma import medgemma_ai
from tts_service import tts_service
from translation_service import translation_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedCallService:
    def __init__(self):
        self.medgemma = medgemma_ai
        self.tts = tts_service
    
    def generate_and_send_ivr_call(self, patient_id: int, schedule_item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IVR script, convert to TTS, and send via Twilio with result tracking"""
        start_time = time.time()
        call_result = {
            "success": False,
            "status": "failed",
            "duration": 0,
            "message": "",
            "error": None
        }
        
        try:
            # Get patient information
            db = SessionLocal()
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            db.close()
            
            if not patient:
                logger.error(f"Patient {patient_id} not found")
                call_result["error"] = "Patient not found"
                return call_result
            
            # Extract patient data from patient record
            patient_name = patient.name
            risk_factors = patient.risk_factors.split(", ") if patient.risk_factors else []
            topic = schedule_item.get("topic", "general")
            message = schedule_item.get("message", "")
            
            # Extract gestational age from diagnosis
            gestational_age = 0
            if patient.diagnosis:
                import re
                ga_match = re.search(r"Week (\d+)", patient.diagnosis)
                if ga_match:
                    gestational_age = int(ga_match.group(1))
            
            # Generate RAG-enhanced script using actual patient data
            logger.info(f"Generating RAG-enhanced script for {patient_name} - {topic}")
            enhanced_script = self.medgemma.generate_medical_script_with_rag(
                topic=topic,
                patient_name=patient_name,
                gestational_age=gestational_age,
                risk_factors=risk_factors
            )

            # Translate to Hindi
            logger.info(f"Translating script to Hindi for {patient_name}")
            hindi_script = translation_service.to_hindi(enhanced_script)

            # Convert to TTS in Hindi
            logger.info(f"Converting Hindi script to TTS for {patient_name}")
            audio_file = self.tts.text_to_speech(hindi_script, language="hi")

            if not audio_file:
                logger.error(f"Failed to generate TTS for {patient_name}")
                call_result["error"] = "TTS generation failed"
                return call_result

            # Send via Twilio using patient's phone number with message option
            logger.info(f"Sending IVR call to {patient_name} at {patient.phone}")

            # Use the new TwiML with message option in Hindi
            twiml = twilio_call_service.create_twiml_with_message_option(hindi_script, patient_id, language="hi-IN")
            success = twilio_call_service._make_call_with_retry(patient.phone, twiml, f"ivr_{patient_id}_{int(time.time())}")

            # Cleanup audio file
            self.tts.cleanup_audio_file(audio_file)
            
            # Calculate call duration
            call_duration = time.time() - start_time
            
            if success.get("success", False):
                logger.info(f"Successfully sent IVR call to {patient_name}")
                call_result.update({
                    "success": True,
                    "status": "completed",
                    "duration": call_duration,
                    "message": f"Call completed successfully for {patient_name}"
                })
            else:
                logger.error(f"Failed to send IVR call to {patient_name}")
                call_result.update({
                    "status": "failed",
                    "duration": call_duration,
                    "message": f"Call failed for {patient_name}"
                })
            
            # Update patient metrics
            self._update_patient_call_metrics(patient_id, call_result)
            
            return call_result
                
        except Exception as e:
            logger.error(f"Error in generate_and_send_ivr_call: {e}")
            call_result.update({
                "error": str(e),
                "duration": time.time() - start_time
            })
            # Update patient metrics even for failed calls
            self._update_patient_call_metrics(patient_id, call_result)
            return call_result
    
    def _update_patient_call_metrics(self, patient_id: int, call_result: Dict[str, Any]):
        """Update patient call metrics in the database"""
        try:
            db = SessionLocal()
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            
            if not patient:
                logger.error(f"Patient {patient_id} not found for metrics update")
                return
            
            # Update call counts
            patient.total_calls_scheduled = (patient.total_calls_scheduled or 0) + 1
            
            status = call_result.get("status", "failed")
            if status == "completed":
                patient.total_calls_completed = (patient.total_calls_completed or 0) + 1
            elif status == "failed":
                patient.total_calls_failed = (patient.total_calls_failed or 0) + 1
            else:
                patient.total_calls_missed = (patient.total_calls_missed or 0) + 1
            
            # Update duration metrics
            duration = call_result.get("duration", 0)
            if duration > 0:
                current_total = patient.total_call_duration or 0
                completed_calls = patient.total_calls_completed or 0
                
                new_total = current_total + duration
                patient.total_call_duration = new_total
                
                if completed_calls > 0:
                    patient.average_call_duration = new_total / completed_calls
            
            # Update last call info
            patient.last_call_date = datetime.now()
            patient.last_call_status = status
            
            # Update call history
            call_history = []
            if patient.call_history:
                try:
                    call_history = json.loads(patient.call_history) if isinstance(patient.call_history, str) else patient.call_history
                except:
                    call_history = []
            
            call_record = {
                "date": datetime.now().isoformat(),
                "status": status,
                "duration": duration,
                "message": call_result.get("message", ""),
                "topic": call_result.get("topic", "")
            }
            call_history.append(call_record)
            patient.call_history = json.dumps(call_history)
            
            # Calculate new success rate
            total_calls = patient.total_calls_scheduled
            completed_calls = patient.total_calls_completed
            if total_calls > 0:
                patient.call_success_rate = (completed_calls / total_calls) * 100
            
            db.commit()
            logger.info(f"Updated call metrics for patient {patient.name}: {status}")
            
        except Exception as e:
            logger.error(f"Error updating patient metrics: {e}")
            db.rollback()
        finally:
            db.close()
    
    def process_scheduled_calls(self):
        """Process all scheduled calls for the current time"""
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            current_time_str = current_time.strftime("%H:%M")
            
            logger.info(f"Processing scheduled calls for {current_date} at {current_time_str}")
            
            # TEMPORARILY DISABLED - Skip processing to prevent errors
            logger.info("Automated call processing temporarily disabled to prevent errors")
            return
            
            # Get all patients
            db = SessionLocal()
            patients = db.query(Patient).all()
            db.close()
            
            for patient in patients:
                if not patient.call_schedule:
                    logger.info(f"Skipping patient {patient.name} - no call schedule")
                    continue
                
                try:
                    logger.info(f"Processing patient {patient.name} with schedule type: {type(patient.call_schedule)}")
                    
                    # Handle different data types
                    schedule_data = patient.call_schedule
                    
                    # If it's a string, try to parse as JSON
                    if isinstance(schedule_data, str):
                        try:
                            schedule_data = json.loads(schedule_data)
                            logger.info(f"Successfully parsed JSON for {patient.name}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON for {patient.name}: {e}")
                            continue
                    
                    # Handle different schedule formats - now expecting consistent dictionary format
                    schedule_items = []
                    
                    if isinstance(schedule_data, dict):
                        if "schedule" in schedule_data:
                            schedule_items = schedule_data["schedule"]
                            logger.info(f"Using dict with 'schedule' key for {patient.name}, {len(schedule_items)} items")
                        else:
                            # Single item dict
                            schedule_items = [schedule_data]
                            logger.info(f"Using single dict item for {patient.name}")
                    elif isinstance(schedule_data, list):
                        schedule_items = schedule_data
                        logger.info(f"Using list format for {patient.name}, {len(schedule_items)} items")
                    else:
                        logger.error(f"Unknown schedule format for {patient.name}: {type(schedule_data)}")
                        continue
                    
                    # Ensure schedule_items is a list
                    if not isinstance(schedule_items, list):
                        logger.error(f"Schedule items is not a list for {patient.name}: {type(schedule_items)}")
                        continue
                    
                    # Process each schedule item
                    for i, item in enumerate(schedule_items):
                        try:
                            logger.info(f"Processing item {i} for {patient.name}: {type(item)}")
                            
                            # Handle different item formats
                            if isinstance(item, dict):
                                item_date_str = item.get("date", "")
                                item_time = item.get("time", "")
                                item_topic = item.get("topic", "general")
                                item_message = item.get("message", "")
                            elif isinstance(item, list) and len(item) >= 2:
                                item_date_str = str(item[0]) if item[0] else ""
                                item_time = str(item[1]) if item[1] else ""
                                item_topic = str(item[2]) if len(item) > 2 else "general"
                                item_message = str(item[3]) if len(item) > 3 else ""
                            else:
                                logger.error(f"Unknown item format for {patient.name} item {i}: {type(item)}")
                                continue
                            
                            if not item_date_str:
                                logger.info(f"Skipping item {i} for {patient.name} - no date")
                                continue
                                
                            try:
                                item_date = datetime.strptime(item_date_str, "%Y-%m-%d").date()
                            except ValueError:
                                logger.error(f"Invalid date format for {patient.name} item {i}: {item_date_str}")
                                continue
                            
                            # Check if this call should be made now
                            if (item_date == current_date and 
                                item_time == current_time_str):
                                
                                logger.info(f"Triggering scheduled call for {patient.name}: {item_topic}")
                                
                                # Create schedule item in dictionary format
                                schedule_item = {
                                    "topic": item_topic,
                                    "message": item_message,
                                    "time": item_time,
                                    "date": item_date_str
                                }
                                
                                # Generate and send the call using patient data
                                call_result = self.generate_and_send_ivr_call(patient.id, schedule_item)
                                
                                if call_result["success"]:
                                    logger.info(f"Successfully processed call for {patient.name}")
                                else:
                                    logger.error(f"Failed to process call for {patient.name}: {call_result.get('error', 'Unknown error')}")
                            else:
                                logger.info(f"Skipping call for {patient.name} - not scheduled for current time")
                        
                        except Exception as e:
                            logger.error(f"Error processing item {i} for {patient.name}: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing patient {patient.name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in process_scheduled_calls: {e}")

# Global instance
automated_call_service = AutomatedCallService() 