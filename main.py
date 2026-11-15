from chestnut.utils.cli import ChestnutCLI
from chestnut.utils.trainer import Trainer


def main():
    cli = ChestnutCLI(
        trainer_class=Trainer,
        save_config_kwargs={
            "overwrite": True,
        },
    )

if __name__ == "__main__":
    main()