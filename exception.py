class GenericException(Exception):
    status_code = 500

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self, message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload


class NotFoundException(GenericException):
    status_code = 404

    def __init__(self, message, payload=None):
        GenericException.__init__(self, message, status_code=NotFoundException.status_code, payload=payload)


class AbortException(GenericException):
    status_code = 400

    def __init__(self, message, payload=None):
        GenericException.__init__(self, message, status_code=AbortException.status_code, payload=payload)
