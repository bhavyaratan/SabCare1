import os
import logging
import yaml
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from datetime import datetime
import requests
from typing import Dict, Any, Optional
from tts_service import tts_service
import html
from twilio.base.exceptions import TwilioException

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_twilio_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)['twilio']

class TwilioCallService:
    def __init__(self):
        self.config = load_twilio_config()
        self.client = None
        self.call_history = {}  # Track call history and status
        self.retry_attempts = 3
        self.retry_delay = 30  # seconds
        
        # Initialize Twilio client if credentials are available
        if self.config['account_sid'] != "YOUR_TWILIO_ACCOUNT_SID":
            try:
                self.client = Client(self.config['account_sid'], self.config['auth_token'])
                logger.info("Twilio client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
    
    def format_phone_number(self, phone_number: str) -> str:
        """Format phone number to E.164 format"""
        import re
        digits = re.sub(r'\D', '', phone_number)
        if len(digits) == 10:
            return f'+1{digits}'
        elif len(digits) == 11 and digits.startswith('1'):
            return f'+{digits}'
        elif digits.startswith('+'):
            return digits
        else:
            raise ValueError(f'Invalid phone number: {phone_number}')
    
    def make_call_and_play_script(self, phone_number: str, script: str, call_type: str = "ivr", 
                                 language: str = "en", retry_on_failure: bool = True) -> Dict[str, Any]:
        """
        Enhanced call function with status tracking and retry logic
        """
        call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{phone_number[-4:]}"
        
        try:
            # Check if using placeholder credentials
            if self.config['account_sid'] == "YOUR_TWILIO_ACCOUNT_SID":
                logger.info(f"[Twilio DEBUG] Using placeholder credentials - call would be made to {phone_number}")
                logger.info(f"[Twilio DEBUG] Script: {script}")
                
                # Simulate call status tracking
                self.track_call_status(call_id, "simulated", {
                    "phone_number": phone_number,
                    "script": script,
                    "call_type": call_type,
                    "language": language
                })
                
                return {
                    "success": True, 
                    "call_id": call_id,
                    "message": "Call simulated (placeholder credentials)",
                    "status": "simulated"
                }
            
            if not self.client:
                raise Exception("Twilio client not initialized")
            
            # Format phone number
            formatted_phone = self.format_phone_number(phone_number)
            
            # Generate TTS audio file
            audio_file = tts_service.text_to_speech(script, language=language)
            
            if not audio_file:
                raise Exception("Failed to generate TTS audio")
            
            # Create TwiML with audio file
            twiml = self._create_twiml_with_audio(audio_file, script)
            
            # Make the call with retry logic
            call_result = self._make_call_with_retry(formatted_phone, twiml, call_id)
            
            # Cleanup audio file
            tts_service.cleanup_audio_file(audio_file)
            
            return call_result
            
        except Exception as e:
            logger.error(f"Error in make_call_and_play_script: {e}")
            
            # Track failed call
            self.track_call_status(call_id, "failed", {
                "error": str(e),
                "phone_number": phone_number,
                "call_type": call_type
            })
            
            return {
                "success": False,
                "call_id": call_id,
                "error": str(e),
                "status": "failed"
            }
    
    def _create_twiml_with_audio(self, audio_file: str, fallback_script: str) -> str:
        """Create TwiML with audio file and fallback TTS"""
        try:
            # For now, we'll use TTS as fallback since Twilio audio file integration requires webhook
            safe_script = html.escape(fallback_script)
            return f'<Response><Say voice="alice">{safe_script}</Say></Response>'
        except Exception as e:
            logger.error(f"Error creating TwiML: {e}")
            return f'<Response><Say voice="alice">Hello, this is your health reminder.</Say></Response>'
    
    def create_twiml_with_message_option(self, script: str, patient_id: int = None, language: str = "en-US") -> str:
        """Create TwiML that plays the script and then offers to leave a message"""
        try:
            safe_script = html.escape(script)

            # Create TwiML with the main script in specified language, then offer to leave a message
            twiml = f'''<Response>
                <Say voice="alice" language="{language}">{safe_script}</Say>
                <Pause length="1"/>
                <Say voice="alice">Press 1 if you'd like to leave a message for our medical team.</Say>
                <Gather input="dtmf" timeout="10" action="/twilio/handle_message_choice" method="POST">
                    <Say voice="alice">Press 1 to leave a message, or hang up to end the call.</Say>
                </Gather>
                <Say voice="alice">Thank you for calling. Goodbye.</Say>
            </Response>'''

            return twiml
        except Exception as e:
            logger.error(f"Error creating TwiML with message option: {e}")
            return f'<Response><Say voice="alice">Hello, this is your health reminder. Goodbye.</Say></Response>'
    
    def create_message_recording_twiml(self, patient_id: int = None) -> str:
        """Create TwiML for recording patient messages"""
        try:
            twiml = f'''<Response>
                <Say voice="alice">Please leave your message after the beep. Press any key when you're done.</Say>
                <Record 
                    action="/twilio/process_message" 
                    method="POST"
                    maxLength="120"
                    timeout="10"
                    playBeep="true"
                    trim="trim-silence">
                </Record>
                <Say voice="alice">Thank you for your message. We'll get back to you soon.</Say>
            </Response>'''
            
            return twiml
        except Exception as e:
            logger.error(f"Error creating message recording TwiML: {e}")
            return f'<Response><Say voice="alice">Thank you for calling. Goodbye.</Say></Response>'
    
    def _make_call_with_retry(self, phone_number: str, twiml: str, call_id: str) -> Dict[str, Any]:
        """Make call with retry logic for failed calls"""
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Making call attempt {attempt + 1} to {phone_number}")
                
                call = self.client.calls.create(
                    to=phone_number,
                    from_=self.config['from_number'],
                    twiml=twiml,
                    status_callback=f"https://your-webhook-url.com/twilio/status/{call_id}",
                    status_callback_event=['initiated', 'ringing', 'answered', 'completed', 'failed', 'busy', 'no-answer']
                )
                
                # Track successful call
                self.track_call_status(call_id, "initiated", {
                    "twilio_sid": call.sid,
                    "phone_number": phone_number,
                    "attempt": attempt + 1
                })
                
                logger.info(f"Call initiated successfully! Call SID: {call.sid}")
                return {
                    "success": True,
                    "call_id": call_id,
                    "twilio_sid": call.sid,
                    "status": "initiated"
                }
                
            except TwilioException as e:
                logger.error(f"Twilio error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_attempts - 1:
                    import time
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_attempts - 1:
                    import time
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
        
        # If all retries failed
        raise Exception(f"All {self.retry_attempts} call attempts failed")
    
    def track_call_status(self, call_id: str, status: str, details: Dict[str, Any] = None):
        """Track call status and delivery confirmations"""
        if call_id not in self.call_history:
            self.call_history[call_id] = {
                "created_at": datetime.now().isoformat(),
                "status_history": []
            }
        
        status_entry = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.call_history[call_id]["status_history"].append(status_entry)
        self.call_history[call_id]["current_status"] = status
        
        logger.info(f"Call {call_id} status updated: {status}")
    
    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """Get call status and delivery confirmation"""
        return self.call_history.get(call_id, {"status": "unknown"})
    
    def get_call_history(self, phone_number: str = None, limit: int = 50) -> list:
        """Get call history filtered by phone number"""
        history = []
        for call_id, call_data in self.call_history.items():
            if phone_number:
                # Check if this call was to the specified phone number
                for status_entry in call_data.get("status_history", []):
                    if status_entry.get("details", {}).get("phone_number") == phone_number:
                        history.append({
                            "call_id": call_id,
                            **call_data
                        })
                        break
            else:
                history.append({
                    "call_id": call_id,
                    **call_data
                })
        
        # Sort by creation time and limit results
        history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return history[:limit]
    
    def handle_missed_calls(self, phone_number: str, original_script: str, 
                           call_type: str = "ivr", max_retries: int = 2) -> bool:
        """Handle missed calls with retry logic"""
        try:
            # Check if we've already retried too many times
            call_history = self.get_call_history(phone_number)
            retry_count = 0
            
            for call in call_history:
                if call.get("current_status") in ["failed", "busy", "no-answer"]:
                    retry_count += 1
            
            if retry_count >= max_retries:
                logger.warning(f"Max retries reached for {phone_number}")
                return False
            
            # Wait before retry
            import time
            time.sleep(60)  # Wait 1 minute before retry
            
            # Retry the call
            result = self.make_call_and_play_script(
                phone_number, 
                original_script, 
                call_type, 
                retry_on_failure=False  # Don't retry the retry
            )
            
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Error handling missed call for {phone_number}: {e}")
            return False
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get call statistics and delivery rates"""
        total_calls = len(self.call_history)
        successful_calls = 0
        failed_calls = 0
        missed_calls = 0
        
        for call_data in self.call_history.values():
            current_status = call_data.get("current_status", "unknown")
            if current_status in ["completed", "answered"]:
                successful_calls += 1
            elif current_status in ["failed", "busy", "no-answer"]:
                failed_calls += 1
                if current_status in ["busy", "no-answer"]:
                    missed_calls += 1
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "missed_calls": missed_calls,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            "delivery_rate": ((successful_calls + missed_calls) / total_calls * 100) if total_calls > 0 else 0
        }

# Global instance
twilio_call_service = TwilioCallService()

def make_callback_call(phone_number: str, callback_message: str, patient_id: int, message_id: int):
    """Make a callback call to a patient with their processed message"""
    try:
        from db import SessionLocal, PatientMessage
        from datetime import datetime
        
        # Initialize Twilio service
        twilio_service = TwilioCallService()
        
        # Create TwiML for the callback
        twiml = f'''<Response>
            <Say voice="alice">{callback_message}</Say>
            <Say voice="alice">If you have any follow-up questions, please call us back. Thank you.</Say>
        </Response>'''
        
        # Make the call
        call_result = twilio_service._make_call_with_retry(phone_number, twiml, f"callback_{message_id}")
        
        # Update message status
        db = SessionLocal()
        patient_message = db.query(PatientMessage).filter(PatientMessage.id == message_id).first()
        if patient_message:
            patient_message.status = "completed"
            patient_message.callback_completed_at = datetime.now()
            db.commit()
        db.close()
        
        logger.info(f"Callback call completed for patient {patient_id}, message {message_id}")
        return call_result
        
    except Exception as e:
        logger.error(f"Error making callback call: {e}")
        return {"success": False, "error": str(e)}

def make_call_and_play_script(phone_number: str, script: str):
    """Legacy function for backward compatibility"""
    service = TwilioCallService()
    return service.make_call_and_play_script(phone_number, script) 