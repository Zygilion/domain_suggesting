# Experiments and analysis

In `domain_suggestions.ipynb` experiments can be found for relevant domain name suggestions based on business description with different prompt engineering techniques, fine tuning open source LLM model and comparing the results.

# Domain Suggestion API

A Fast API based REST API that generates domain name suggestions based on business descriptions using a fine-tuned Qwen3-0.6B language model.

## Prerequisites

- Python >=3.11
- At least 2GB VRAM or RAM if not running on GPU

## Running the API

### Install dependencies and run python script
```bash
pip install -r requirements.txt
```


```bash
python main.py
```

The API will start on `http://localhost:8000`

## API Integration Guide

### 1. API Endpoint Details

**Endpoint**: `POST /generate-domains`

**Content-Type**: `application/json`

### 2. Request Format

Send a POST request with the following JSON structure:

```json
{
  "description": "Interior design studio specializing in modern minimalist homes"
}
```

**Field Specifications**:
- `description` (string, required): Business description to generate domain suggestions for
- Maximum length: 1000 characters
- Should be descriptive and include key business aspects

### 3. Response Format

The API returns a JSON response with the following structure:

```json
{
  "suggestions": [
    "modernminimalistdesign.studio",
    "cleanlineinteriors.com",
    "minimalistdesignlab.co",
    "simplespacedesign.net",
    "modernhomedesigners.org",
    "minimalinteriorpro.biz",
    "sleekspacesstudio.info",
    "contemporarydesignhub.design",
    "minimalistlivingdesign.store",
    "modernspacecreators.consulting"
  ],
  "processing_time": 2.34,
  "business_description": "Interior design studio specializing in modern minimalist homes"
}
```

There should always be 10 unique suggestions

**Response Fields**:
- `suggestions` (array): List of generated domain suggestions
- `processing_time` (float): Time taken to generate suggestions in seconds
- `business_description` (string): Echo of the input description

### 4. Error Handling

**HTTP Status Codes**:
- `200`: Success
- `400`: Bad request (invalid input format or validation errors)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

**Error Response Format**:
```json
{
  "detail": "Error message description"
}
```

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc


## Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/generate-domains" \
     -H "Content-Type: application/json" \
     -d '{
       "description": "An eco-friendly coffee shop in downtown Seattle"
     }'
```
