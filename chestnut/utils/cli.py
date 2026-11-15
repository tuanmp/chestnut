from lightning.pytorch.cli import LightningCLI


class ChestnutCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        if lr_scheduler is None:
            return optimizer
        config = {
            "scheduler": lr_scheduler,
            "interval": getattr(lr_scheduler, "interval", "epoch"),
            "frequency": getattr(lr_scheduler, "frequency", 1),
        }
        monitor = getattr(lr_scheduler, "monitor", None)
        if monitor is not None:
            config["monitor"] = monitor
        return {"optimizer": optimizer, "lr_scheduler": config}