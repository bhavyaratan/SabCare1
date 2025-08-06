import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import os
from datetime import datetime, timedelta
import json
import re
from typing import Dict, Any, List, Optional
from rag_service import rag_service

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# When generating or copying any 'time' field, always use 'h:mm AM/PM' format
# For example, if med_time is '9 AM', convert to '9:00 AM'
def ensure_time_format(time_str):
    import re
    match = re.match(r'^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)$', time_str, re.IGNORECASE)
    if match:
        hour = match.group(1)
        minute = match.group(2) if match.group(2) else '00'
        ampm = match.group(3).upper()
        return f"{int(hour)}:{minute} {ampm}"
    return time_str

class MedGemmaAI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load a medical-specialized model for healthcare applications"""
        try:
            logger.info("Loading medical AI model...")
            
            # Load quantized Gemma model from Unsloth
            model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"

            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Medical AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading medical model: {e}")
            # Fallback to a very simple approach
            self._load_simple_fallback()
    
    def _load_simple_fallback(self):
        """Load a very simple fallback if model loading fails"""
        try:
            logger.info("Loading simple fallback...")
            # Create a simple tokenizer and model
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Simple fallback loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            raise
    
    def generate_comprehensive_ivr_schedule(self, gestational_age_weeks: int, patient_name: str, current_date: datetime = None, risk_factors: list = None, risk_category: str = "low", structured_medications: list = None) -> dict:
        """Generate comprehensive IVR schedule with specific dates and detailed messages based on risk factors and call frequency"""
        if current_date is None:
            current_date = datetime.now()
        if risk_factors is None:
            risk_factors = []
        if structured_medications is None:
            structured_medications = []
        schedule = []

        # --- Medication Reminders ---
        for med in structured_medications:
            med_name = med.get("name", "your medication")
            med_time = ensure_time_format(med.get("time") or "09:00 AM")
            med_days = med.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            med_dosage = med.get("dosage", "as prescribed")
            med_frequency = med.get("frequency", None)  # e.g., "daily", "weekly", etc.
            # For each day in med_days, schedule reminders for each week until week 40
            for week_offset in range(0, max(0, 40 - gestational_age_weeks) + 1):
                week_num = gestational_age_weeks + week_offset
                if week_num > 40:
                    break
                for day in med_days:
                    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
                    weekday = day_map.get(day, 0)
                    # Calculate the date for this week and day
                    week_start = current_date + timedelta(weeks=week_offset)
                    days_ahead = weekday - week_start.weekday()
                    if days_ahead < 0:
                        days_ahead += 7
                    reminder_date = week_start + timedelta(days=days_ahead)
                    schedule.append({
                        "type": "medication_reminder",
                        "date": reminder_date.strftime("%Y-%m-%d"),
                        "time": med_time,
                        "topic": f"Medication Reminder - {med_name}",
                        "message": f"This is your reminder to take {med_name} ({med_dosage}) at {med_time}.",
                        "week": week_num
                    })

        # --- Check-in/Call Frequency by Risk Category ---
        # Generate check-in/call reminders from current week to week 40
        current_week = gestational_age_weeks
        week_offset = 0
        
        while current_week + week_offset <= 40:
            week_num = current_week + week_offset
            call_date = current_date + timedelta(weeks=week_offset)
            
            if risk_category == "high":
                # High risk: Twice weekly (Monday and Thursday)
                # Calculate Monday and Thursday of the current week
                monday_offset = (0 - call_date.weekday()) % 7  # Days until Monday
                thursday_offset = (3 - call_date.weekday()) % 7  # Days until Thursday
                
                monday_date = call_date + timedelta(days=monday_offset)
                thursday_date = call_date + timedelta(days=thursday_offset)
                
                # Add Monday call
                schedule.append({
                    "week": int(week_num),
                    "date": monday_date.strftime("%Y-%m-%d"),
                    "time": ensure_time_format("10:00 AM"),
                    "topic": f"Week {int(week_num)} check-in (Monday)",
                    "message": self._generate_weekly_checkin_message(patient_name, int(week_num), risk_factors),
                    "type": "checkin"
                })
                
                # Add Thursday call
                schedule.append({
                    "week": int(week_num),
                    "date": thursday_date.strftime("%Y-%m-%d"),
                    "time": ensure_time_format("3:30 PM"),
                    "topic": f"Week {int(week_num)} check-in (Thursday)",
                    "message": self._generate_weekly_checkin_message(patient_name, int(week_num), risk_factors),
                    "type": "checkin"
                })
                week_offset += 1  # Move to next week
            elif risk_category == "medium":
                # Medium risk: Weekly
                schedule.append({
                    "week": int(week_num),
                    "date": call_date.strftime("%Y-%m-%d"),
                    "time": ensure_time_format("3:30 PM"),
                    "topic": f"Week {int(week_num)} check-in",
                    "message": self._generate_weekly_checkin_message(patient_name, int(week_num), risk_factors),
                    "type": "checkin"
                })
                week_offset += 1  # Move to next week
            else:
                # Low risk: Biweekly (every 2 weeks)
                schedule.append({
                    "week": int(week_num),
                    "date": call_date.strftime("%Y-%m-%d"),
                    "time": ensure_time_format("3:30 PM"),
                    "topic": f"Week {int(week_num)} check-in",
                    "message": self._generate_weekly_checkin_message(patient_name, int(week_num), risk_factors),
                    "type": "checkin"
                })
                week_offset += 2  # Move to next biweekly interval

        # Add special milestone messages (as before)
        # First trimester (weeks 1-12)
        if gestational_age_weeks <= 12:
            schedule.extend([
                {
                    "week": 8,
                    "date": (current_date + timedelta(weeks=8-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("10:00 AM"),
                    "topic": "Early pregnancy care and nutrition",
                    "message": self._generate_week_8_message(patient_name, risk_factors)
                },
                {
                    "week": 10,
                    "date": (current_date + timedelta(weeks=10-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("2:00 PM"),
                    "topic": "First trimester screening preparation",
                    "message": self._generate_week_10_message(patient_name, risk_factors)
                },
                {
                    "week": 12,
                    "date": (current_date + timedelta(weeks=12-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("11:00 AM"),
                    "topic": "End of first trimester milestone",
                    "message": self._generate_week_12_message(patient_name, risk_factors)
                }
            ])
        
        # Second trimester (weeks 13-28)
        elif gestational_age_weeks <= 28:
            schedule.extend([
                {
                    "week": 16,
                    "date": (current_date + timedelta(weeks=16-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("9:00 AM"),
                    "topic": "Anatomy scan preparation",
                    "message": self._generate_week_16_message(patient_name, risk_factors)
                },
                {
                    "week": 20,
                    "date": (current_date + timedelta(weeks=20-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("3:00 PM"),
                    "topic": "Gender reveal and movement tracking",
                    "message": self._generate_week_20_message(patient_name, risk_factors)
                },
                {
                    "week": 24,
                    "date": (current_date + timedelta(weeks=24-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("10:30 AM"),
                    "topic": "Gestational diabetes screening",
                    "message": self._generate_week_24_message(patient_name, risk_factors)
                },
                {
                    "week": 28,
                    "date": (current_date + timedelta(weeks=28-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("2:30 PM"),
                    "topic": "Third trimester preparation",
                    "message": self._generate_week_28_message(patient_name, risk_factors)
                }
            ])
        
        # Third trimester (weeks 29-40)
        else:
            schedule.extend([
                {
                    "week": 32,
                    "date": (current_date + timedelta(weeks=32-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("11:00 AM"),
                    "topic": "Birth plan and labor signs",
                    "message": self._generate_week_32_message(patient_name, risk_factors)
                },
                {
                    "week": 36,
                    "date": (current_date + timedelta(weeks=36-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("9:30 AM"),
                    "topic": "Group B strep testing and final preparations",
                    "message": self._generate_week_36_message(patient_name, risk_factors)
                },
                {
                    "week": 38,
                    "date": (current_date + timedelta(weeks=38-gestational_age_weeks)).strftime("%Y-%m-%d"),
                    "time": ensure_time_format("2:00 PM"),
                    "topic": "Full-term pregnancy and labor preparation",
                    "message": self._generate_week_38_message(patient_name, risk_factors)
                }
            ])
        
        # Filter out past dates and sort by date
        current_date_only = current_date.date()
        schedule = [item for item in schedule if datetime.strptime(item["date"], "%Y-%m-%d").date() >= current_date_only]
        schedule.sort(key=lambda x: x["date"])
        
        # Convert datetime objects to strings for JSON serialization
        for item in schedule:
            if isinstance(item["date"], datetime):
                item["date"] = item["date"].strftime("%Y-%m-%d")
        
        logger.info(f"Generated schedule before deduplication: {schedule}")
        # Remove duplicate week check-ins for low-risk patients (after all schedule.extend calls)
        if risk_category == "low":
            seen = set()
            unique_schedule = []
            for item in schedule:
                if item.get("type") == "checkin" or ("week" in item and "date" in item and "time" in item):
                    key = (item.get("week"), item.get("date"), item.get("time"))
                    if key in seen:
                        continue
                    seen.add(key)
                unique_schedule.append(item)
            schedule = unique_schedule
        logger.info(f"Schedule after deduplication: {schedule}")
        
        return {"success": True, "schedule": schedule, "gestational_age": gestational_age_weeks, "patient_name": patient_name, "risk_factors": risk_factors or []}
    
    def _generate_week_8_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 8 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 8-week pregnancy update. At this stage, your baby's major organs are developing. Focus on healthy eating with plenty of folic acid, iron, and calcium. Take your prenatal vitamins daily and stay hydrated. Avoid alcohol, smoking, and raw fish. Schedule your first prenatal appointment if you haven't already. Remember to get plenty of rest and light exercise like walking."
        
        # Add risk factor-specific advice
        if "diabetes" in risk_factors:
            base_message += " Since you have diabetes, monitor your blood sugar closely and follow your doctor's dietary recommendations. You may need more frequent prenatal visits."
        if "hypertension" in risk_factors:
            base_message += " With your history of high blood pressure, monitor your blood pressure regularly and report any sudden increases to your healthcare provider immediately."
        if "advanced_maternal_age" in risk_factors:
            base_message += " As an older mother, you may be offered additional screening tests. Discuss these options with your healthcare provider."
        if "smoking" in risk_factors:
            base_message += " If you're still smoking, please work with your healthcare provider to quit. Smoking significantly increases pregnancy risks."
        if "severe_underweight" in risk_factors:
            base_message += " Your low BMI requires special attention. You need to gain adequate weight during pregnancy - aim for 28-40 pounds total. Eat frequent, nutrient-dense meals including protein, healthy fats, and complex carbohydrates. Consider working with a nutritionist to ensure proper weight gain. Monitor for signs of malnutrition like fatigue, dizziness, or hair loss."
        if "underweight" in risk_factors:
            base_message += " Your BMI indicates you're underweight. Focus on healthy weight gain during pregnancy - aim for 25-35 pounds total. Eat nutrient-rich foods including lean proteins, whole grains, and healthy fats. Consider smaller, more frequent meals if you have trouble eating large portions."
        if "obesity" in risk_factors:
            base_message += " Your BMI indicates obesity, which can increase pregnancy risks. Work closely with your healthcare provider to manage weight gain - you may need to gain less weight than normal (11-20 pounds). Monitor for gestational diabetes and preeclampsia. Focus on healthy eating and safe exercise. You may need more frequent prenatal visits and additional monitoring."
        if "overweight" in risk_factors:
            base_message += " Your BMI indicates you're overweight. Focus on healthy weight gain during pregnancy - aim for 15-25 pounds total. Monitor for gestational diabetes and maintain a balanced diet with regular exercise. Work with your healthcare provider to manage weight appropriately."
        
        base_message += " Contact your healthcare provider if you experience severe nausea, bleeding, or abdominal pain."
        return base_message
    
    def _generate_week_10_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 10 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 10-week pregnancy reminder. Your first trimester screening is coming up. This includes blood tests and possibly an ultrasound to check for chromosomal abnormalities. The screening is optional but recommended."
        
        if "advanced_maternal_age" in risk_factors:
            base_message += " Given your age, additional genetic screening may be recommended. Discuss these options with your healthcare provider."
        if "diabetes" in risk_factors:
            base_message += " Your diabetes screening will be more comprehensive. Continue monitoring your blood sugar levels closely."
        
        base_message += " You may experience morning sickness, fatigue, and breast tenderness - these are normal. Continue taking prenatal vitamins and eating small, frequent meals. Stay hydrated and get adequate sleep. If morning sickness is severe, talk to your doctor about safe medications."
        return base_message
    
    def _generate_week_12_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 12 message based on risk factors"""
        base_message = f"Hello {patient_name}, congratulations! You've completed your first trimester at 12 weeks. Your risk of miscarriage has significantly decreased. Your baby is now about the size of a lime with all major organs formed. You should start feeling better as morning sickness typically improves. Your energy levels should increase. Schedule your second trimester appointments."
        
        if "diabetes" in risk_factors:
            base_message += " Continue monitoring your blood sugar and follow your diabetes management plan closely."
        if "hypertension" in risk_factors:
            base_message += " Keep monitoring your blood pressure and report any concerning readings to your healthcare provider."
        
        base_message += " Consider starting pregnancy exercise classes and childbirth education. Remember to continue healthy eating and prenatal vitamins."
        return base_message
    
    def _generate_week_16_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 16 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 16-week pregnancy update. Your anatomy scan is scheduled soon - this is an exciting milestone where you can see your baby's development in detail. The scan checks for proper organ development and can reveal the baby's gender if you want to know. Drink plenty of water before the scan for better imaging. You should start feeling baby movements soon, called 'quickening.' Your uterus is now about the size of a grapefruit."
        
        if "diabetes" in risk_factors:
            base_message += " Your anatomy scan will include additional monitoring for diabetes-related complications. Continue monitoring your blood sugar levels."
        if "hypertension" in risk_factors:
            base_message += " The scan will also check for any signs of preeclampsia. Continue monitoring your blood pressure regularly."
        if "advanced_maternal_age" in risk_factors:
            base_message += " Additional screening may be recommended due to your age. Discuss these options with your healthcare provider."
        
        base_message += " Continue with prenatal vitamins and healthy eating. Consider starting to plan your maternity leave and childcare arrangements."
        return base_message
    
    def _generate_week_20_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 20 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 20-week pregnancy check-in. You're halfway through your pregnancy! You should definitely be feeling baby movements by now. Track these movements daily - you should feel at least 10 movements in 2 hours. If you notice decreased movement, contact your healthcare provider immediately. Your baby is now about the size of a banana."
        
        if "diabetes" in risk_factors:
            base_message += " Continue monitoring your blood sugar closely. Your healthcare provider may recommend more frequent ultrasounds to monitor baby's growth."
        if "hypertension" in risk_factors:
            base_message += " Monitor your blood pressure regularly and report any sudden increases. Your doctor may recommend additional monitoring."
        if "multiple_pregnancy" in risk_factors:
            base_message += " With twins or multiples, you'll need more frequent monitoring and may deliver earlier than expected."
        
        base_message += " You may experience round ligament pain as your uterus grows. Consider starting pregnancy yoga or swimming for gentle exercise. Begin thinking about your birth plan and whether you want pain relief during labor."
        return base_message
    
    def _generate_week_24_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 24 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 24-week pregnancy update. Your gestational diabetes screening is scheduled. This test checks how your body processes sugar during pregnancy. You'll need to fast for the test and drink a glucose solution. The test takes about 3 hours."
        
        if "diabetes" in risk_factors:
            base_message += " Since you have diabetes, this screening is especially important. You may need additional monitoring throughout your pregnancy."
        if "obesity" in risk_factors:
            base_message += " Your risk of gestational diabetes is higher, so this screening is particularly important."
        
        base_message += " If diagnosed with gestational diabetes, you'll need to monitor your blood sugar and possibly adjust your diet. Your baby is now about the size of an ear of corn and weighs about 1.5 pounds. You may experience back pain and swelling in your feet. Consider wearing a pregnancy support belt and elevating your feet when possible."
        return base_message
    
    def _generate_week_28_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 28 message based on risk factors"""
        base_message = f"Hello {patient_name}, welcome to your third trimester at 28 weeks! You're entering the final stretch of your pregnancy. Your baby is now about the size of an eggplant and weighs about 2.5 pounds. You'll have more frequent prenatal visits now - every 2 weeks, then weekly after 36 weeks. Your baby's brain is developing rapidly."
        
        if "diabetes" in risk_factors:
            base_message += " Continue monitoring your blood sugar closely. You may need more frequent ultrasounds to monitor baby's growth and amniotic fluid levels."
        if "hypertension" in risk_factors:
            base_message += " Monitor your blood pressure regularly and watch for signs of preeclampsia: sudden swelling, headaches, or vision changes."
        if "previous_cesarean" in risk_factors:
            base_message += " Since you've had a previous cesarean, discuss your birth plan with your healthcare provider. You may be a candidate for VBAC or may need a repeat cesarean."
        
        base_message += " You may experience Braxton Hicks contractions, which are normal practice contractions. Start preparing for labor by taking childbirth classes. Pack your hospital bag and install your car seat. Consider writing a birth plan and discussing it with your healthcare provider."
        return base_message
    
    def _generate_week_32_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 32 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 32-week pregnancy update. Your baby is now about the size of a large jicama and weighs about 4 pounds. Your baby's bones are hardening, except for the skull which remains soft for birth. You should be having weekly prenatal visits now. Learn to recognize the signs of labor: regular contractions, water breaking, bloody show, or lower back pain."
        
        if "diabetes" in risk_factors:
            base_message += " Continue monitoring your blood sugar and watch for signs of macrosomia (large baby). Your healthcare provider may discuss induction if the baby is getting too large."
        if "hypertension" in risk_factors:
            base_message += " Monitor your blood pressure closely and report any sudden increases. Watch for signs of preeclampsia."
        if "multiple_pregnancy" in risk_factors:
            base_message += " With twins or multiples, you may deliver earlier than expected. Be prepared for possible preterm labor."
        
        base_message += " Create your birth plan including pain management preferences. Your baby should be head-down by now. If not, your doctor may suggest exercises to encourage proper positioning. Start sleeping on your left side for better blood flow."
        return base_message
    
    def _generate_week_36_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 36 message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your 36-week pregnancy check-in. Your Group B strep test is scheduled - this is routine for all pregnancies. Your baby is now about the size of a head of romaine lettuce and weighs about 6 pounds. Your baby is considered 'late preterm' and would likely do well if born now."
        
        if "diabetes" in risk_factors:
            base_message += " Your healthcare provider may recommend induction at 39 weeks to reduce risks associated with diabetes."
        if "hypertension" in risk_factors:
            base_message += " Continue monitoring your blood pressure. Your doctor may recommend induction if your blood pressure becomes difficult to control."
        if "previous_cesarean" in risk_factors:
            base_message += " If you're planning a VBAC, discuss the risks and benefits with your healthcare provider. You may need to deliver in a hospital with surgical capabilities."
        
        base_message += " You may experience increased pressure in your pelvis as the baby 'drops' or engages. Your cervix may begin to dilate and efface. Have your hospital bag packed and car seat installed. Know the route to your hospital and have emergency contacts ready. Your baby's lungs are nearly fully developed."
        return base_message
    
    def _generate_week_38_message(self, patient_name: str, risk_factors: list) -> str:
        """Generate personalized week 38 message based on risk factors"""
        base_message = f"Hello {patient_name}, congratulations! Your baby is now considered full-term at 38 weeks. Your baby is about the size of a leek and weighs about 7 pounds. Your baby's brain is still developing rapidly. You may experience more frequent Braxton Hicks contractions. These are normal and help prepare your body for labor."
        
        if "diabetes" in risk_factors:
            base_message += " Your healthcare provider may recommend induction soon to reduce risks associated with diabetes."
        if "hypertension" in risk_factors:
            base_message += " Monitor your blood pressure closely. Your doctor may recommend induction if your blood pressure becomes difficult to control."
        if "previous_cesarean" in risk_factors:
            base_message += " If you're planning a VBAC, be aware that labor may progress differently than expected. Stay in close communication with your healthcare provider."
        
        base_message += " Watch for signs of labor: regular contractions that get stronger and closer together, water breaking, or bloody show. If your water breaks, go to the hospital immediately. You may lose your mucus plug, which is normal. Stay hydrated and rest when possible. Your baby could arrive any day now!"
        return base_message
    
    def _generate_weekly_checkin_message(self, patient_name: str, week: int, risk_factors: list) -> str:
        """Generate personalized weekly check-in message based on risk factors"""
        base_message = f"Hello {patient_name}, this is your week {week} pregnancy check-in. How are you feeling today? Remember to track your baby's movements - you should feel at least 10 movements in 2 hours. If you notice decreased movement, contact your healthcare provider immediately."
        
        # Add nutrition-specific guidance based on risk factors
        if "severe_underweight" in risk_factors:
            base_message += " Continue focusing on healthy weight gain. Are you eating enough calories and protein? Track your weight gain weekly and report any concerns to your healthcare provider. Consider protein-rich snacks between meals."
        elif "underweight" in risk_factors:
            base_message += " Monitor your weight gain progress. Are you eating nutrient-dense foods regularly? Consider adding healthy snacks between meals to boost your calorie intake."
        elif "obesity" in risk_factors:
            base_message += " Continue monitoring your weight and blood sugar levels. Are you maintaining a healthy diet and safe exercise routine? Report any unusual symptoms like swelling or headaches immediately."
        elif "overweight" in risk_factors:
            base_message += " Keep monitoring your weight gain and maintain healthy eating habits. Are you staying active with safe pregnancy exercises?"
        
        # Add other risk factor guidance
        if "diabetes" in risk_factors:
            base_message += " Continue monitoring your blood sugar levels closely and follow your diabetes management plan."
        if "hypertension" in risk_factors:
            base_message += " Monitor your blood pressure regularly and report any sudden increases to your healthcare provider."
        if "advanced_maternal_age" in risk_factors:
            base_message += " Given your age, be extra vigilant about any unusual symptoms and report them promptly."
        
        base_message += " Continue with your prenatal vitamins and healthy eating. Stay hydrated and get adequate rest. Don't hesitate to call your doctor if you have any concerns about your pregnancy. You're doing great!"
        return base_message

    def process_medical_query(self, query: str, patient_name: str = "", context: str = "") -> str:
        """Process medical queries using the AI model"""
        try:
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            # Create medical context prompt
            medical_prompt = f"Medical Assistant: Hello {patient_name}, I'm your AI health companion. Patient asks: {query} Medical Assistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(medical_prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Medical Assistant:" in response:
                response = response.split("Medical Assistant:")[-1].strip()
            
            # Add medical disclaimers
            response += "\n\nNote: This is AI-generated medical information. Always consult your healthcare provider for personalized medical advice."
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing medical query: {e}")
            return f"Hello {patient_name}, I'm having trouble processing your medical query. Please contact your healthcare provider directly for assistance."
    
    def extract_medical_info(self, text: str) -> dict:
        """Extract medical information from text using the AI model"""
        try:
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            # Create extraction prompt
            extraction_prompt = f"Extract medical information from: {text} Patient Name:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(extraction_prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # Generate extraction
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse extracted information
            extracted_info = self._parse_medical_extraction(response)
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting medical info: {e}")
            return {
                "patient_name": "",
                "gestational_age_weeks": 0,
                "due_date": "",
                "medications": [],
                "risk_factors": [],
                "race": "",
                "height": "",
                "weight": "",
                "age": 0
            }
    
    def _parse_medical_extraction(self, response: str) -> dict:
        """Parse extracted medical information with improved name extraction and risk factors"""
        
        extracted_info = {
            "patient_name": "",
            "gestational_age_weeks": 0,
            "due_date": "",
            "medications": [],
            "risk_factors": [],
            "race": "",
            "height": "",
            "weight": "",
            "age": 0
        }
        
        # Improved patient name extraction - look for patterns like "Patient: Name" or "Name, X weeks"
        name_patterns = [
            r"Patient Name:\s*([A-Za-z\s]+?)(?:\n|$)",
            r"Patient[:\s]+([A-Za-z\s]+?)(?:,|\s+\d+|\s+weeks?)",
            r"([A-Za-z\s]+?),\s*\d+\s*weeks?\s*pregnant",
            r"Patient[:\s]+([A-Za-z\s]+?)(?:\s|$)",
            r"([A-Za-z\s]+?)\s+\d+\s*weeks?\s*pregnant"
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, response, re.IGNORECASE)
            if name_match:
                extracted_info["patient_name"] = name_match.group(1).strip()
                break
        
        # Extract gestational age
        ga_match = re.search(r"Gestational Age:\s*(\d+)\s*weeks?", response, re.IGNORECASE)
        if ga_match:
            extracted_info["gestational_age_weeks"] = int(ga_match.group(1))
        
        # Extract due date
        due_match = re.search(r"Due Date:\s*([^\n]+)", response, re.IGNORECASE)
        if due_match:
            extracted_info["due_date"] = due_match.group(1).strip()
        
        # Extract medications
        med_matches = re.findall(r"Medications:\s*([^\n]+)", response, re.IGNORECASE)
        if med_matches:
            extracted_info["medications"] = [med.strip() for med in med_matches[0].split(",")]
        
        # Extract structured medications with time, dosage, and frequency
        # First, find all medication patterns in the text
        structured_med_pattern = r"([A-Za-z\s]+?)\s*-\s*(\d{1,2})\s*(AM|PM)\s*\(([^)]+)\)(?:\s*-\s*([^,\n]+))?"
        structured_med_matches = re.findall(structured_med_pattern, response, re.IGNORECASE)
        if structured_med_matches:
            extracted_info["structured_medications"] = []
            for match in structured_med_matches:
                med_name = match[0].strip()
                time = match[1]
                ampm = match[2]
                days = [day.strip() for day in match[3].split(', ')]
                dosage = match[4].strip() if match[4] else "As prescribed"
                extracted_info["structured_medications"].append({
                    "name": med_name,
                    "time": f"{time} {ampm}",
                    "days": days,
                    "dosage": dosage
                })
        
        # Extract race
        race_patterns = [
            r"Race:\s*([^\n]+)",
            r"race[:\s]+([A-Za-z\s]+)",
            r"ethnicity[:\s]+([A-Za-z\s]+)",
            r"([A-Za-z]+)\s+race"
        ]
        for pattern in race_patterns:
            race_match = re.search(pattern, response, re.IGNORECASE)
            if race_match:
                extracted_info["race"] = race_match.group(1).strip()
                break
        
        # Extract height
        height_patterns = [
            r"Height:\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)",
            r"height[:\s]+(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)",
            r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)\s*height",
            r"height[:\s]+(\d+(?:\.\d+)?)"
        ]
        for pattern in height_patterns:
            height_match = re.search(pattern, response, re.IGNORECASE)
            if height_match:
                extracted_info["height"] = height_match.group(1)
                break
        
        # Extract weight
        weight_patterns = [
            r"Weight:\s*(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)",
            r"weight[:\s]+(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)",
            r"(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\s*weight",
            r"weight[:\s]+(\d+(?:\.\d+)?)"
        ]
        for pattern in weight_patterns:
            weight_match = re.search(pattern, response, re.IGNORECASE)
            if weight_match:
                extracted_info["weight"] = weight_match.group(1)
                break
        
        # Extract age
        age_patterns = [
            r"Age:\s*(\d+)\s*years?",
            r"Age:\s*(\d+)",
            r"(\d+)\s*years?\s*old",
            r"age\s+(\d+)"
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, response, re.IGNORECASE)
            if age_match:
                extracted_info["age"] = int(age_match.group(1))
                break
        
        # Extract risk factors
        risk_factors = []
        
        # Check for common pregnancy risk factors
        risk_patterns = [
            (r"diabetes|gestational diabetes", "diabetes"),
            (r"hypertension|high blood pressure", "hypertension"),
            (r"obesity|overweight", "obesity"),
            (r"advanced maternal age|age\s+\d+", "advanced_maternal_age"),
            (r"previous\s+preterm|preterm\s+history", "preterm_history"),
            (r"multiple\s+pregnancy|twins|triplets", "multiple_pregnancy"),
            (r"placenta\s+previa", "placenta_previa"),
            (r"preeclampsia", "preeclampsia"),
            (r"smoking|tobacco", "smoking"),
            (r"alcohol|drinking", "alcohol_use"),
            (r"anemia|low\s+iron", "anemia"),
            (r"thyroid|hypothyroidism|hyperthyroidism", "thyroid_disorder"),
            (r"asthma", "asthma"),
            (r"depression|anxiety|mental\s+health", "mental_health"),
            (r"previous\s+cesarean|c-section", "previous_cesarean"),
            # Enhanced nutrition-related risk factors
            (r"eating\s+disorder|anorexia|bulimia", "eating_disorder"),
            (r"malnutrition|undernutrition", "malnutrition"),
            (r"vitamin\s+deficiency|vitamin\s+d|vitamin\s+b12|folate\s+deficiency", "vitamin_deficiency"),
            (r"severe\s+underweight|very\s+thin|extremely\s+thin", "severe_underweight"),
            (r"underweight|low\s+weight|thin", "underweight"),
            (r"morbid\s+obesity|severe\s+obesity", "severe_obesity"),
            (r"binge\s+eating|compulsive\s+eating", "binge_eating"),
            (r"food\s+restriction|dieting|calorie\s+restriction", "food_restriction"),
            (r"pica|eating\s+non-food\s+items", "pica"),
            (r"gastroparesis|slow\s+digestion", "gastroparesis"),
            (r"celiac|gluten\s+sensitivity", "celiac_disease"),
            (r"food\s+allergies|severe\s+allergies", "food_allergies")
        ]
        
        for pattern, risk_factor in risk_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                risk_factors.append(risk_factor)
        
        extracted_info["risk_factors"] = risk_factors
        
        return extracted_info
    
    def generate_medical_script(self, topic: str, patient_name: str, gestational_age: int = 0) -> str:
        """Generate medical voice script using the AI model"""
        try:
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            # Create medical script prompt
            script_prompt = f"Generate a medical voice script for patient {patient_name}, {gestational_age} weeks pregnant, topic: {topic} Script:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(script_prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # Generate script
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract script
            if "Script:" in response:
                script = response.split("Script:")[-1].strip()
            else:
                script = response
            
            # Format for voice
            formatted_script = f"Hello {patient_name}, this is your AI health companion calling with your {gestational_age}-week pregnancy update. {script}"
            
            return formatted_script
            
        except Exception as e:
            logger.error(f"Error generating medical script: {e}")
            return f"Hello {patient_name}, this is your pregnancy health reminder. Please contact your healthcare provider for personalized advice."

    def generate_medical_script_with_rag(self, topic: str, patient_name: str, gestational_age: int = 0, risk_factors: list = None) -> str:
        """Generate medical script using RAG-enhanced guidelines"""
        # Generate base message
        base_message = self.generate_medical_script(topic, patient_name, gestational_age)
        
        # Enhance with RAG guidelines
        enhanced_message = rag_service.enhance_message_with_guidelines(
            base_message, topic, risk_factors
        )
        
        return enhanced_message

    def calculate_bmi_and_risk(self, height_cm: str, weight_kg: str, age: int, risk_factors: list) -> dict:
        """Calculate BMI and categorize patient risk level with enhanced nutrition monitoring"""
        try:
            # Convert height and weight to numbers with better error handling
            try:
                height = float(height_cm) if height_cm and str(height_cm).strip() else 0
            except (ValueError, TypeError):
                height = 0
                logger.warning(f"Could not convert height '{height_cm}' to float, using 0")
            
            try:
                weight = float(weight_kg) if weight_kg and str(weight_kg).strip() else 0
            except (ValueError, TypeError):
                weight = 0
                logger.warning(f"Could not convert weight '{weight_kg}' to float, using 0")
            
            # Calculate BMI
            bmi = 0
            if height > 0 and weight > 0:
                height_m = height / 100  # Convert cm to meters
                bmi = weight / (height_m * height_m)
            
            # Enhanced nutrition monitoring
            nutrition_status = "normal"
            nutrition_concerns = []
            
            # Detailed BMI categorization for nutrition monitoring
            if bmi < 16:
                nutrition_status = "severely_underweight"
                nutrition_concerns.append("severe_underweight")
                nutrition_concerns.append("nutritional_deficiency")
            elif bmi < 18.5:
                nutrition_status = "underweight"
                nutrition_concerns.append("underweight")
                nutrition_concerns.append("nutritional_risk")
            elif bmi >= 25 and bmi < 30:
                nutrition_status = "overweight"
                nutrition_concerns.append("overweight")
            elif bmi >= 30:
                nutrition_status = "obese"
                nutrition_concerns.append("obesity")
            else:
                nutrition_status = "normal"
            
            # Determine risk category based on BMI, age, and risk factors
            risk_score = 0
            
            # Enhanced BMI risk factors with nutrition focus
            if bmi < 16:
                risk_score += 4  # Severely underweight - high risk
            elif bmi < 18.5:
                risk_score += 2  # Underweight - moderate risk
            elif bmi >= 25 and bmi < 30:
                risk_score += 2  # Overweight
            elif bmi >= 30:
                risk_score += 3  # Obese
            elif bmi >= 18.5 and bmi < 25:
                risk_score += 0  # Normal weight
            
            # Age risk factors
            if age >= 35:
                risk_score += 2  # Advanced maternal age
            elif age >= 30:
                risk_score += 1  # Moderate age risk
            
            # Medical risk factors
            if "diabetes" in risk_factors:
                risk_score += 3
            if "hypertension" in risk_factors:
                risk_score += 3
            if "obesity" in risk_factors:
                risk_score += 2
            if "advanced_maternal_age" in risk_factors:
                risk_score += 2
            if "previous_cesarean" in risk_factors:
                risk_score += 2
            if "preterm_history" in risk_factors:
                risk_score += 3
            if "multiple_pregnancy" in risk_factors:
                risk_score += 3
            if "placenta_previa" in risk_factors:
                risk_score += 4
            if "preeclampsia" in risk_factors:
                risk_score += 4
            if "smoking" in risk_factors:
                risk_score += 2
            if "alcohol_use" in risk_factors:
                risk_score += 2
            if "anemia" in risk_factors:
                risk_score += 1
            if "thyroid_disorder" in risk_factors:
                risk_score += 2
            if "asthma" in risk_factors:
                risk_score += 1
            if "mental_health" in risk_factors:
                risk_score += 1
            
            # Additional nutrition-related risk factors
            if "eating_disorder" in risk_factors:
                risk_score += 3
                nutrition_concerns.append("eating_disorder")
            if "malnutrition" in risk_factors:
                risk_score += 2
                nutrition_concerns.append("malnutrition")
            if "vitamin_deficiency" in risk_factors:
                risk_score += 1
                nutrition_concerns.append("vitamin_deficiency")
            
            # Categorize risk level
            if risk_score <= 2:
                risk_category = "low"
                call_frequency = "biweekly"
            elif risk_score <= 5:
                risk_category = "medium"
                call_frequency = "weekly"
            else:
                risk_category = "high"
                call_frequency = "twice_weekly"
            
            # Special handling for severely underweight patients
            if bmi < 16:
                risk_category = "high"  # Force high risk for severely underweight
                call_frequency = "twice_weekly"
            
            return {
                "bmi": round(bmi, 1) if bmi > 0 else 0,
                "risk_category": risk_category,
                "call_frequency": call_frequency,
                "risk_score": risk_score,
                "risk_factors_count": len(risk_factors),
                "nutrition_status": nutrition_status,
                "nutrition_concerns": nutrition_concerns
            }
            
        except Exception as e:
            logger.error(f"Error calculating BMI and risk: {e}")
            return {
                "bmi": 0,
                "risk_category": "low",
                "call_frequency": "biweekly",
                "risk_score": 0,
                "risk_factors_count": 0,
                "nutrition_status": "unknown",
                "nutrition_concerns": []
            }

    def generate_postnatal_care_schedule(self, patient_name: str, delivery_date: datetime, current_date: datetime = None, delivery_type: str = "vaginal") -> dict:
        """Generate postnatal care schedule for the first month after childbirth"""
        if current_date is None:
            current_date = datetime.now()
        
        schedule = []
        
        # Calculate weeks postpartum
        days_postpartum = (current_date - delivery_date).days
        weeks_postpartum = min(days_postpartum // 7, 4)  # Max 4 weeks
        
        # Generate weekly postnatal care messages
        for week in range(1, 5):  # Weeks 1-4 postpartum
            week_date = delivery_date + timedelta(weeks=week)
            
            # Only schedule if the week hasn't passed yet
            if week_date > current_date:
                message = self._generate_postnatal_week_message(patient_name, week, delivery_type)
                
                schedule.append({
                    "type": "postnatal_care",
                    "week": week,
                    "date": week_date.strftime("%Y-%m-%d"),
                    "time": "10:00 AM",
                    "topic": f"Week {week} Postnatal Care",
                    "message": message
                })
        
        return {
            "success": True,
            "schedule": schedule,
            "total_weeks": 4,
            "delivery_date": delivery_date.strftime("%Y-%m-%d"),
            "delivery_type": delivery_type
        }
    
    def _generate_postnatal_week_message(self, patient_name: str, week: int, delivery_type: str) -> str:
        """Generate personalized postnatal care message for specific week"""
        
        if week == 1:
            base_message = f"Hello {patient_name}, welcome to your first week postpartum! Your body is healing from childbirth. Expect vaginal bleeding (lochia), uterine contractions, and breast engorgement. Rest frequently and stay hydrated."
            
            if delivery_type == "c-section":
                base_message += " Care for your incision: keep it clean and dry, avoid heavy lifting, and watch for signs of infection like redness or fever."
            else:
                base_message += " Care for your perineum: use warm water for cleaning, apply ice packs for swelling, and take sitz baths for comfort."
            
            base_message += " Contact your healthcare provider immediately if you experience fever, heavy bleeding, or severe pain. You're doing great!"
            
        elif week == 2:
            base_message = f"Hello {patient_name}, you're in week 2 of your postpartum recovery. Your bleeding should be decreasing and your energy may be improving slightly."
            
            base_message += " Focus on gentle walking and pelvic floor exercises. Sleep when your baby sleeps, even during the day. Continue monitoring for signs of infection or postpartum depression."
            
            if delivery_type == "c-section":
                base_message += " Your incision should be healing well. Continue avoiding heavy lifting and strenuous activity."
            
            base_message += " Schedule your postpartum checkup for 2-6 weeks after delivery. You're making excellent progress!"
            
        elif week == 3:
            base_message = f"Hello {patient_name}, welcome to week 3 of your postpartum journey. You may notice your hair starting to fall out - this is normal due to hormonal changes."
            
            base_message += " Continue bonding with your baby through skin-to-skin contact, talking, and responding to their cues. Your emotional state should be stabilizing, but contact your healthcare provider if you're still experiencing severe sadness or anxiety."
            
            base_message += " If you're breastfeeding, it should be getting easier. Contact a lactation consultant if you're still having difficulties. You're doing wonderfully!"
            
        elif week == 4:
            base_message = f"Hello {patient_name}, congratulations on reaching week 4 postpartum! You're approaching the end of your initial recovery period."
            
            base_message += " Your postpartum checkup should be scheduled soon. Your doctor will check your physical recovery, discuss birth control options, and address any concerns."
            
            base_message += " You may be ready to start gentle exercise like walking and pelvic floor exercises. Listen to your body and stop if you experience pain or fatigue."
            
            base_message += " Remember, every woman's recovery is different. Trust your instincts and don't hesitate to ask for help when needed. You've made it through the first month!"
        
        return base_message
    
    def generate_postnatal_medical_script(self, topic: str, patient_name: str, postpartum_week: int = 1, delivery_type: str = "vaginal") -> str:
        """Generate medical scripts for postnatal care topics"""
        
        if topic.lower() == "breastfeeding":
            return f"Hello {patient_name}, this is your breastfeeding support call. Feed your baby 8-12 times per day, ensuring proper latch to prevent nipple soreness. Watch for hunger cues and ensure your baby is getting enough milk with 6+ wet diapers daily. Contact a lactation consultant if you experience pain or difficulty."
        
        elif topic.lower() == "postpartum_depression":
            return f"Hello {patient_name}, this is your mental health check-in. Watch for signs of postpartum depression: persistent sadness, difficulty bonding with baby, changes in appetite or sleep, or thoughts of self-harm. Baby blues are normal for 1-2 weeks, but severe symptoms require immediate medical attention. You're not alone - help is available."
        
        elif topic.lower() == "physical_recovery":
            if delivery_type == "c-section":
                return f"Hello {patient_name}, this is your C-section recovery check-in. Keep your incision clean and dry, avoid soaking in water for 2 weeks, and don't lift anything heavier than your baby. Watch for signs of infection: redness, swelling, foul odor, or fever. Contact your doctor immediately if you notice these symptoms."
            else:
                return f"Hello {patient_name}, this is your physical recovery check-in. Your vaginal bleeding should be decreasing. Use warm water for perineal care, take sitz baths for comfort, and continue pelvic floor exercises. Contact your doctor for heavy bleeding, large clots, or foul odor."
        
        elif topic.lower() == "nutrition":
            return f"Hello {patient_name}, this is your postpartum nutrition reminder. Focus on protein-rich foods, whole grains, fruits, vegetables, and healthy fats. Stay hydrated with water and electrolyte drinks. Include iron-rich foods to replenish blood loss. Small, frequent meals may be easier to manage than large meals."
        
        else:
            return f"Hello {patient_name}, this is your postnatal care check-in. Remember to rest when possible, stay hydrated, and contact your healthcare provider if you have any concerns about your recovery. You're doing great!"

# Global MedGemma instance
medgemma_ai = MedGemmaAI() 