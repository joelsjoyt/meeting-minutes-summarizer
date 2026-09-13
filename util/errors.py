class TranscriptionError(Exception):
    pass

class TranscriptionPipelineError(Exception):
    pass

class SummarizationModelError(Exception):
    pass

class SummarizationPipelineError(Exception):
    pass

class PDFGenerationError(Exception):
    pass

class ModelDownloadError(Exception):
    pass

class ModelDownloadPipelineError(Exception):
    pass

class LLMResponseCleanerError(Exception):
    pass

class DeviceError(Exception):
    pass

class ApplicationError(Exception):
    pass

class ApplicationUIError(Exception):
    pass
