import logging


class SampleClass:
    """"
    Add description of:
        1. what object class represents
        2. what are its inputs/params
    """
    def __init__(self, input_file):
        self.input_file = input_file

    def read_data(self):
        # add description only if not self-explainatory
        try:
            # do something
            logging.info("Some message")
        except Exception as e:
            logging.error(f"read_data error: {e}")

    def run(self):
        # Read Data
        self.read_data()


if __name__ == '__main__':
    input_file = 'input_file'
    sc = SampleClass(input_file)
    sc.run()


