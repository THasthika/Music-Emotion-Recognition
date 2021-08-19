"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import importlib.util
import time
import torch

# from ..utils.util import list_classes_from_module, load_label_mapping

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
	   First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        logger.debug("Loading torchscript model")
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """
        return torch.jit.load(model_pt_path, map_location=self.device)

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        audio_data = data['audio'] if 'audio' in data else None
        lyrics_data = data['lyrics'] if 'lyrics' in data else None

        data = torch.rand((1, 4))

        return torch.as_tensor(data, device=self.device)

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data, *args, **kwargs)
        return results

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        return data.tolist()

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        data_preprocess = self.preprocess(data)

        if not self._is_explain():
            output = self.inference(data_preprocess)
            output = self.postprocess(output)
        else:
            output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output

    def explain_handle(self, data_preprocess, raw_data):
        """Captum explanations handler

        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request

        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = None
        inputs = None
        target = 0

        logger.info("Calculating Explanations")
        row = raw_data[0]
        if isinstance(row, dict):
            logger.info("Getting data and target")
            inputs = row.get("data") or row.get("body")
            target = row.get("target")
            if not target:
                target = 0

        output_explain = self.get_insights(data_preprocess, inputs, target)
        return output_explain

    def _is_explain(self):
        if self.context and self.context.get_request_header(0, "explain"):
            if self.context.get_request_header(0, "explain") == "True":
                self.explain = True
                return True
        return False