from pathlib import Path
import pandas as pd
from cpp_bag.plot import AnnoMark, plot_embedding

def draw_embedding(df_p, suffix="", write_pdf=False, write_html=False):
    plot_df = pd.read_json(df_p, orient="records")
    fig = plot_embedding(
        plot_df,
        marks=[
            AnnoMark("19_0487_AS", "A"),
            AnnoMark("18_0118_AS", "B"),
            AnnoMark("18_0175_AS", "C"),
            AnnoMark("18_0188_AS", "D"),
        ],
    )
    parent_dir = Path(df_p).parent.parent
    if write_html:
        fig.write_html((parent_dir / f"embedding{suffix}.html").absolute())
    if write_pdf:
        fig.write_image(
            (parent_dir / f"embedding{suffix}.pdf").absolute(), format="pdf"
        )


if __name__ == "__main__":
    # MODEL_P = "experiments0/trial2/pool-1659028139226.pth"
    # model = BagPooling.from_checkpoint(MODEL_P)
    # model.eval()
    # empty_arr = np.expand_dims(np.concatenate([np.expand_dims(np.load("data/empty_cell_feat.npy"), 0) for _ in range(257)]), 0) 
    # print(empty_arr.shape)
    # with torch.no_grad():
    #     out_distribute = model(torch.as_tensor(empty_arr)).numpy()[0]

    # measure_slide_vectors(
    #     "experiments0/trial2/train2_pool.pkl",
    #     "experiments0/trial2/val2_pool.pkl",
    #     dummy_baseline=False,
    #     out_distribute=out_distribute,
    #     dst=Path("OUT"),
    # )

    draw_embedding("experiments2/trial0/pool0.json", suffix="_pool", write_pdf=True)
    draw_embedding("experiments2/trial0/avg0.json", suffix="_avg", write_pdf=True)
