from dotenv import load_dotenv
from sifaka.models import create_model

# Load environment variables from .env file
load_dotenv()


def test_create_model_with_groq():

    production_quality_models = [
        "groq:gemma2-9b-it",
        "groq:meta-llama/llama-guard-4-12b"
        "groq:llama-3.3-70b-versatile",
        "groq:llama-3.1-8b-instant",
        "groq:llama-3.3-70b-8192",
        "groq:llama-3.3-8b-8192",        
        "whisper-large-v3"
    ]
    model = create_model(production_quality_models[0])
    assert model is not None
    assert model.model_name == production_quality_models[0]
    model = create_model(production_quality_models[1])
    assert model is not None
    assert model.model_name == production_quality_models[1]
    model = create_model(production_quality_models[2])
    assert model is not None
    assert model.model_name == production_quality_models[2]
    model = create_model(production_quality_models[3])
    assert model is not None
    assert model.model_name == production_quality_models[3]
    model = create_model(production_quality_models[4])
    assert model is not None
    assert model.model_name == production_quality_models[4]
    model = create_model(production_quality_models[5])
    assert model is not None
    assert model.model_name == production_quality_models[5]
