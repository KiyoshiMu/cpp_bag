from pathlib import Path
import pandas as pd
from cpp_bag.plot import plot_embedding 


def draw_embedding(df_p, suffix="", write_pdf=False, write_html=False):
    plot_df = pd.read_json(df_p, orient="records")
    fig = plot_embedding(plot_df)
    parent_dir = Path(df_p).parent.parent
    if write_html:
        fig.write_html((parent_dir / f"embedding{suffix}.html").absolute())
    if write_pdf:
        fig.write_image((parent_dir / f"embedding{suffix}.pdf").absolute(), format="pdf")
        
if __name__ == "__main__":

    draw_embedding("experiments0/trial2/pool2.json", suffix="_pool", write_pdf=True)
    draw_embedding("experiments0/trial2/avg2.json", suffix="_avg", write_pdf=True)