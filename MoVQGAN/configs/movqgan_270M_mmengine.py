from movqgan.models.vqgan import MOVQ

model = dict(
    type=MOVQ,
    learning_rate=1e-4,
    ema_decay=0.9999,
    embed_dim=4,
    n_embed=16384,
    monitor='val/rec_loss',
    ddconfig=dict(
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=256,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
    ),
    lossconfig=dict(
        target='movqgan.modules.losses.vqperceptual.VQLPIPSWithDiscriminator2',
        params=dict(
            disc_conditional=False,
            disc_in_channels=3,
            disc_num_layers=2,
            disc_start=1,
            disc_weight=0.8,
            codebook_weight=1.0,
        ),
    ),
)
