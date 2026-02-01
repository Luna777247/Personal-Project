"""
Triton gRPC Client
High-performance inference client using gRPC protocol
"""

import time
from typing import Dict, List, Optional, Union, Any
import numpy as np

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from loguru import logger


class TritonGRPCClient:
    """gRPC client for Triton Inference Server"""
    
    def __init__(
        self,
        url: str = "localhost:8001",
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: Optional[str] = None,
        private_key: Optional[str] = None,
        certificate_chain: Optional[str] = None
    ):
        """
        Initialize Triton gRPC client
        
        Args:
            url: Triton server URL (host:port)
            verbose: Enable verbose logging
            ssl: Use SSL/TLS connection
            root_certificates: Path to root certificates
            private_key: Path to private key
            certificate_chain: Path to certificate chain
        """
        self.url = url
        self.verbose = verbose
        
        try:
            self.client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=ssl,
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain
            )
            logger.info(f"Connected to Triton Server: {url}")
            
            # Check server health
            if self.client.is_server_live():
                logger.info("✓ Server is live")
            if self.client.is_server_ready():
                logger.info("✓ Server is ready")
                
        except InferenceServerException as e:
            logger.error(f"Failed to connect to Triton: {e}")
            raise
    
    def get_server_metadata(self) -> Dict[str, Any]:
        """Get Triton server metadata"""
        try:
            metadata = self.client.get_server_metadata()
            return {
                "name": metadata.name,
                "version": metadata.version,
                "extensions": metadata.extensions
            }
        except InferenceServerException as e:
            logger.error(f"Failed to get server metadata: {e}")
            raise
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict[str, Any]:
        """
        Get model metadata
        
        Args:
            model_name: Name of the model
            model_version: Version of the model (empty string for latest)
        
        Returns:
            Model metadata dictionary
        """
        try:
            metadata = self.client.get_model_metadata(model_name, model_version)
            
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": inp.shape
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": out.shape
                    }
                    for out in metadata.outputs
                ]
            }
        except InferenceServerException as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise
    
    def get_model_config(self, model_name: str, model_version: str = "") -> Dict[str, Any]:
        """
        Get model configuration
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
        
        Returns:
            Model configuration dictionary
        """
        try:
            config = self.client.get_model_config(model_name, model_version)
            return {
                "name": config.config.name,
                "platform": config.config.platform,
                "max_batch_size": config.config.max_batch_size,
                "input": [
                    {
                        "name": inp.name,
                        "data_type": inp.data_type,
                        "dims": inp.dims
                    }
                    for inp in config.config.input
                ],
                "output": [
                    {
                        "name": out.name,
                        "data_type": out.data_type,
                        "dims": out.dims
                    }
                    for out in config.config.output
                ]
            }
        except InferenceServerException as e:
            logger.error(f"Failed to get model config: {e}")
            raise
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """
        Check if model is ready for inference
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
        
        Returns:
            True if model is ready
        """
        try:
            return self.client.is_model_ready(model_name, model_version)
        except InferenceServerException as e:
            logger.error(f"Failed to check model readiness: {e}")
            return False
    
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        model_version: str = "",
        request_id: str = "",
        timeout: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference
        
        Args:
            model_name: Name of the model
            inputs: Dictionary mapping input names to numpy arrays
            outputs: List of output names to retrieve (None for all)
            model_version: Version of the model
            request_id: Custom request ID for tracking
            timeout: Request timeout in seconds
        
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        try:
            # Prepare inputs
            triton_inputs = []
            for input_name, input_data in inputs.items():
                # Ensure correct shape and type
                if not isinstance(input_data, np.ndarray):
                    input_data = np.array(input_data)
                
                # Create Triton input
                triton_input = grpcclient.InferInput(
                    input_name,
                    input_data.shape,
                    self._numpy_to_triton_dtype(input_data.dtype)
                )
                triton_input.set_data_from_numpy(input_data)
                triton_inputs.append(triton_input)
            
            # Prepare outputs
            triton_outputs = []
            if outputs:
                for output_name in outputs:
                    triton_outputs.append(grpcclient.InferRequestedOutput(output_name))
            
            # Perform inference
            start_time = time.time()
            
            response = self.client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs if triton_outputs else None,
                model_version=model_version,
                request_id=request_id,
                timeout=timeout
            )
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            if self.verbose:
                logger.debug(f"Inference time: {inference_time:.2f}ms")
            
            # Extract outputs
            results = {}
            for output in response.get_response().outputs:
                results[output.name] = response.as_numpy(output.name)
            
            return results
            
        except InferenceServerException as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def infer_batch(
        self,
        model_name: str,
        inputs: Dict[str, List[np.ndarray]],
        outputs: Optional[List[str]] = None,
        model_version: str = "",
        batch_size: int = 32
    ) -> List[Dict[str, np.ndarray]]:
        """
        Perform batch inference with automatic batching
        
        Args:
            model_name: Name of the model
            inputs: Dictionary mapping input names to lists of numpy arrays
            outputs: List of output names to retrieve
            model_version: Version of the model
            batch_size: Maximum batch size
        
        Returns:
            List of result dictionaries
        """
        # Get number of samples
        num_samples = len(list(inputs.values())[0])
        
        # Validate all inputs have same length
        for input_name, input_data in inputs.items():
            if len(input_data) != num_samples:
                raise ValueError(f"Input {input_name} has different length")
        
        results = []
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            # Prepare batch inputs
            batch_inputs = {}
            for input_name, input_data in inputs.items():
                batch_data = np.array(input_data[i:end_idx])
                batch_inputs[input_name] = batch_data
            
            # Infer
            batch_results = self.infer(
                model_name=model_name,
                inputs=batch_inputs,
                outputs=outputs,
                model_version=model_version
            )
            
            results.append(batch_results)
        
        return results
    
    def async_infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        model_version: str = "",
        request_id: str = "",
        callback: Optional[callable] = None
    ):
        """
        Perform asynchronous inference
        
        Args:
            model_name: Name of the model
            inputs: Dictionary mapping input names to numpy arrays
            outputs: List of output names to retrieve
            model_version: Version of the model
            request_id: Custom request ID
            callback: Callback function for results
        """
        # Prepare inputs
        triton_inputs = []
        for input_name, input_data in inputs.items():
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data)
            
            triton_input = grpcclient.InferInput(
                input_name,
                input_data.shape,
                self._numpy_to_triton_dtype(input_data.dtype)
            )
            triton_input.set_data_from_numpy(input_data)
            triton_inputs.append(triton_input)
        
        # Prepare outputs
        triton_outputs = []
        if outputs:
            for output_name in outputs:
                triton_outputs.append(grpcclient.InferRequestedOutput(output_name))
        
        # Async infer
        self.client.async_infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs if triton_outputs else None,
            callback=callback,
            model_version=model_version,
            request_id=request_id
        )
    
    def get_inference_statistics(
        self,
        model_name: str = "",
        model_version: str = ""
    ) -> Dict[str, Any]:
        """
        Get inference statistics for a model
        
        Args:
            model_name: Name of the model (empty for all models)
            model_version: Version of the model
        
        Returns:
            Statistics dictionary
        """
        try:
            stats = self.client.get_inference_statistics(model_name, model_version)
            
            return {
                "model_stats": [
                    {
                        "name": model_stat.name,
                        "version": model_stat.version,
                        "inference_count": model_stat.inference_count,
                        "execution_count": model_stat.execution_count,
                        "success_count": model_stat.success_count,
                        "failure_count": model_stat.failure_count,
                        "queue_time_ns": model_stat.queue_time_ns,
                        "compute_input_time_ns": model_stat.compute_input_time_ns,
                        "compute_infer_time_ns": model_stat.compute_infer_time_ns,
                        "compute_output_time_ns": model_stat.compute_output_time_ns,
                    }
                    for model_stat in stats.model_stats
                ]
            }
        except InferenceServerException as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    def _numpy_to_triton_dtype(self, numpy_dtype) -> str:
        """Convert numpy dtype to Triton data type string"""
        dtype_map = {
            np.bool_: "BOOL",
            np.int8: "INT8",
            np.int16: "INT16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
            np.uint16: "UINT16",
            np.uint32: "UINT32",
            np.uint64: "UINT64",
            np.float16: "FP16",
            np.float32: "FP32",
            np.float64: "FP64",
        }
        
        return dtype_map.get(numpy_dtype.type, "FP32")
    
    def close(self):
        """Close the client connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Client connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = TritonGRPCClient(url="localhost:8001", verbose=True)
    
    # Get server metadata
    server_info = client.get_server_metadata()
    print(f"\nServer: {server_info}")
    
    # Check if model exists
    model_name = "fraud_detector"
    if client.is_model_ready(model_name):
        print(f"\n✓ Model '{model_name}' is ready")
        
        # Get model metadata
        model_info = client.get_model_metadata(model_name)
        print(f"\nModel info: {model_info}")
        
        # Prepare test input
        test_input = np.random.rand(1, 10).astype(np.float32)
        
        # Perform inference
        results = client.infer(
            model_name=model_name,
            inputs={"float_input": test_input}
        )
        
        print(f"\nInference results:")
        for output_name, output_data in results.items():
            print(f"  {output_name}: {output_data}")
    else:
        print(f"\n✗ Model '{model_name}' is not ready")
    
    # Close connection
    client.close()
