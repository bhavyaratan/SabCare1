import logging
from googletrans import Translator

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.translator = Translator()

    def to_hindi(self, text: str) -> str:
        """Translate given text to Hindi."""
        try:
            result = self.translator.translate(text, dest='hi')
            return result.text
        except Exception as e:
            logger.error(f"Hindi translation failed: {e}")
            return text

translation_service = TranslationService()
