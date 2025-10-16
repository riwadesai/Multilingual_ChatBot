"""
LLM Manager Module
Manages free Hugging Face language models for generation
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict
import logging

logger = logging.getLogger(__name__)

class HuggingFaceLLM(LLM):
    """Custom LangChain LLM wrapper for Hugging Face models"""
    
    def __init__(self, model_name: str, max_length: int = 512, 
                 temperature: float = 0.7, device: str = "auto"):
        """
        Initialize Hugging Face LLM
        
        Args:
            model_name: Name of the Hugging Face model
            max_length: Maximum sequence length
            temperature: Sampling temperature
            device: Device to run on
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model"""
        try:
            logger.info(f"Loading LLM model: {self.model_name}")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to CPU-only model
            try:
                logger.info("Trying CPU-only fallback...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    temperature=self.temperature
                )
                logger.info("CPU fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                raise
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, 
              **kwargs) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        try:
            # Generate text
            result = self.pipeline(
                prompt,
                max_length=min(len(prompt.split()) + 100, self.max_length),
                num_return_sequences=1,
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Apply stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return "huggingface"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters"""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "device": self.device
        }

class LLMManager:
    """Manages multiple LLM models and provides unified interface"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize LLM manager
        
        Args:
            model_name: Name of the primary model
        """
        self.model_name = model_name
        self.primary_llm = None
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models"""
        try:
            # Initialize primary model
            self.primary_llm = HuggingFaceLLM(self.model_name)
            self.models['primary'] = self.primary_llm
            
            logger.info(f"Initialized LLM manager with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM manager: {e}")
            raise
    
    def get_llm(self, model_name: str = 'primary') -> HuggingFaceLLM:
        """
        Get LLM instance
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            LLM instance
        """
        return self.models.get(model_name, self.primary_llm)
    
    def generate_response(self, prompt: str, model_name: str = 'primary', 
                         **kwargs) -> str:
        """
        Generate response using specified model
        
        Args:
            prompt: Input prompt
            model_name: Name of the model to use
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        try:
            llm = self.get_llm(model_name)
            return llm(prompt, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def generate_with_template(self, template: str, context: str, 
                             question: str, model_name: str = 'primary') -> str:
        """
        Generate response using a template
        
        Args:
            template: Prompt template
            context: Context information
            question: User question
            model_name: Name of the model to use
            
        Returns:
            Generated response
        """
        try:
            # Format template
            formatted_prompt = template.format(
                context=context,
                question=question
            )
            
            return self.generate_response(formatted_prompt, model_name)
            
        except Exception as e:
            logger.error(f"Failed to generate with template: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def get_model_info(self, model_name: str = 'primary') -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        llm = self.get_llm(model_name)
        if llm:
            return llm._identifying_params
        return {}
    
    def list_models(self) -> List[str]:
        """
        List available models
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def add_model(self, name: str, model_name: str) -> bool:
        """
        Add a new model
        
        Args:
            name: Name to assign to the model
            model_name: Hugging Face model name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            new_llm = HuggingFaceLLM(model_name)
            self.models[name] = new_llm
            logger.info(f"Added model '{name}' with Hugging Face model '{model_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model '{name}': {e}")
            return False
    
    def remove_model(self, name: str) -> bool:
        """
        Remove a model
        
        Args:
            name: Name of the model to remove
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.models and name != 'primary':
            del self.models[name]
            logger.info(f"Removed model '{name}'")
            return True
        return False
