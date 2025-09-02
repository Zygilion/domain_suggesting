import logging
import time
from typing import List, Tuple

from fastapi import HTTPException
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from domain_parser import DomainParser

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the AI model lifecycle and domain generation"""

    def __init__(self, model_name: str = 'Qwen/Qwen3-0.6B', model_path: str = 'fine-tune-model/'):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.fine_tuned_model = None
        self.model_loaded = False
        self.domain_parser = DomainParser()

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info("Loading model and tokenizer...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype='auto',
                device_map='auto'
            )

            # Load fine-tuned model
            self.fine_tuned_model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype='auto',
                is_trainable=False
            )

            self.model_loaded = True
            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def generate_domains(self, business_description: str) -> Tuple[List[str], float]:
        """Generate domain suggestions for a business description"""
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            start_time = time.time()

            # Prepare prompt
            prompt = f'Here is the business description:\n{business_description}'
            messages = [
                {'role': 'user', 'content': prompt}
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors='pt').to(self.fine_tuned_model.device)

            # Generation configuration
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0
            )

            # Generate text
            generated_ids = self.fine_tuned_model.generate(
                **model_inputs,
                generation_config=generation_config,
                max_new_tokens=500
            )

            # Decode output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip('\n')

            processing_time = time.time() - start_time

            # Parse domain suggestions from the generated content
            domain_suggestions = self.domain_parser.parse_domain_suggestions(content)

            return domain_suggestions, processing_time

        except Exception as e:
            logger.error(f"Error generating domains: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating domain suggestions: {str(e)}")

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready"""
        return self.model_loaded