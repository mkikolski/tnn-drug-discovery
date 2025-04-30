from data_access import DataAccess


class Steps:
    @staticmethod
    def fetch_data(**kwargs) -> dict:
        DataAccess.get_general_training_chembl_data("data/general", limit=1000, smi_length=100, tc=2496335)
        return {}

    @staticmethod
    def pretrain_generator(**kwargs) -> dict:
        return {}

    @staticmethod
    def train_dqn(**kwargs) -> dict:
        return {}