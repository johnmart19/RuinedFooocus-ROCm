"""Browser-compatible encoding for silent video pipelines."""

from fractions import Fraction

import av


def save_mp4(path, frames, fps):
    """Encode RGB PIL frames as H.264 without changing their dimensions."""
    if not frames:
        raise ValueError("Cannot save an empty video.")
    with av.open(str(path), mode="w", format="mp4",
                 options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for image in frames:
            if image.size != frames[0].size:
                raise ValueError("Video frames must have matching dimensions.")
            frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
