from abc import ABC, abstractmethod

class ModelPath(ABC):
    @classmethod
    def create_path(cls, args, epoch):
        return cls(args).generate_path(epoch)

    def __init__(self, args):
        self.args = args

    def _get_tune_type(self):
        if self.args.lora:
            return f"lora-{self.args.lora_r}-{self.args.lora_alpha}"
        elif self.args.layer_ids != [1]:
            return f"layer-{self.args.layer_ids[0]}-{self.args.layer_ids[1]}-{self.args.layer_ids[2]}"
        else:
            return "full"

    def _get_model_name(self):
        return self.args.model_name_or_path.split('/')[-1]

    @abstractmethod
    def generate_path(self, epoch):
        """Each subclass implements its own path generation logic"""
        pass

class GAModelPath(ModelPath):
    def generate_path(self, epoch):
        model_name = self._get_model_name()
        tune_type = self._get_tune_type()
        return f"models/{model_name}_{self.args.loss_type}_{tune_type}_beta-{self.args.beta}_lr-{self.args.lr:.0e}__batch-{self.args.batch_size}_epoch-{epoch}"

class RMUModelPath(ModelPath):
    def generate_path(self, epoch):
        model_name = self._get_model_name()
        tune_type = self._get_tune_type()
        return f"models/{model_name}_rmu_{tune_type}_alpha-{int(self.args.alpha[0])}_steer-{self.args.steering_coeffs}_lr-{self.args.lr:.0e}_batch-{self.args.batch_size}_epoch-{epoch}"



class RepNoiseModelPath(ModelPath):
    def generate_path(self, epoch):
            model_name = self._get_model_name()
            tune_type = self._get_tune_type()
            return f"models/{model_name}_repnoise_{tune_type}_alpha-{args.alpha}_beta-{args.beta}_lr-{args.lr:.0e}_batch-{self.args.batch_size}_epoch-{epoch}"



