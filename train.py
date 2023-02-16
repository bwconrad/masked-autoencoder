import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import MIMDataModule
from src.model import MaskedAutoencoderModel
from src.pl_utils import MyLightningArgumentParser, init_logger

model_class = MaskedAutoencoderModel
dm_class = MIMDataModule

# Parse arguments
parser = MyLightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data", skip=["patch_size"])
parser.add_lightning_class_args(model_class, "model")
parser.link_arguments("data.size", "model.image_size")
args = parser.parse_args()

# Setup trainer
logger = init_logger(args)
checkpoint_callback = ModelCheckpoint(
    filename="best-{epoch}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_last=True,
)
model = model_class(**args["model"])
dm = dm_class(patch_size=model.patch_size, **args["data"])

trainer = pl.Trainer.from_argparse_args(
    args, logger=logger, callbacks=[checkpoint_callback]
)

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)
