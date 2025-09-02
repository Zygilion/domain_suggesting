import logging
import time
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Domain Suggestion API",
    description="Generate domain name suggestions based on business description",
)


# Pydantic models for request/response
class BusinessDescription(BaseModel):
    description: str = Field(
        max_length=1000,
        description="Business description to generate domain suggestions for"
    )


class DomainResponse(BaseModel):
    suggestions: List[str]
    processing_time: float
    business_description: str


# Global variables for model components
tokenizer = None
fine_tuned_model = None


class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.fine_tuned_model = None
        self.model_loaded = False

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info("Loading model and tokenizer...")

            model_name = 'Qwen/Qwen3-0.6B'

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype='auto',
                device_map='auto'
            )

            # Load fine-tuned model
            self.fine_tuned_model = PeftModel.from_pretrained(
                base_model,
                'fine-tune-model/',
                torch_dtype='auto',
                is_trainable=False
            )

            self.model_loaded = True
            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def generate_domains(self, business_description: str) -> tuple[List[str], float]:
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
            # This is a simple parser - you might need to adjust based on your model's output format
            domain_suggestions = self._parse_domain_suggestions(content)

            return domain_suggestions, processing_time

        except Exception as e:
            logger.error(f"Error generating domains: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating domain suggestions: {str(e)}")

    def _parse_domain_suggestions(self, content: str) -> List[str]:
        """Parse domain suggestions from model output"""
        # This is a basic parser - adjust based on your model's output format
        lines = content.split('\n')
        domains = []

        for line in lines:
            line = line.strip()
            # Look for lines that might contain domains
            if '.' in line and len(line) > 3:
                # Extract potential domains (basic cleaning)
                words = line.split()
                for word in words:
                    if '.' in word and len(word) > 3:
                        # Basic domain validation
                        if word.count('.') >= 1 and not word.startswith('.') and not word.endswith('.'):
                            domains.append(word.lower())

        # If no domains found in expected format, return the raw content split by common separators
        if not domains:
            # Fallback: split content and look for domain-like strings
            content_clean = content.replace(',', '\n').replace(';', '\n')
            domains = [line.strip() for line in content_clean.split('\n') if line.strip()]

        return domains


model_manager = ModelManager()
model_manager.load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status"""
    return {
        "status": "healthy" if model_manager.model_loaded else "unhealthy",
        "model_loaded": model_manager.model_loaded,
        "timestamp": time.time()
    }


@app.post("/generate-domains", response_model=DomainResponse)
async def generate_domains(request: BusinessDescription):
    """Generate domain suggestions based on business description"""

    if not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the health endpoint.")

    try:
        logger.info(f"Generating domains for: {request.description}")

        suggestions, processing_time = model_manager.generate_domains(
            request.description
        )

        logger.info(f"Generated {len(suggestions)} suggestions in {processing_time:.2f}s")

        return DomainResponse(
            suggestions=suggestions,
            processing_time=processing_time,
            business_description=request.description
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
