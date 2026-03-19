from __future__ import annotations

import base64

import streamlit as st

from app.services.detection_service import DetectionService

st.set_page_config(page_title="Forensic Detector", page_icon="🛰️", layout="wide")


@st.cache_resource
def get_service() -> DetectionService:
    return DetectionService()


def main() -> None:
    st.title("Forensic Detector")
    st.caption("Detect AI-generated or manipulated images with an ensemble forensic pipeline.")

    include_heatmap = st.checkbox("Generate heatmap", value=True)
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])

    if not uploaded:
        st.info("Upload an image to start analysis.")
        return

    file_bytes = uploaded.read()
    if not file_bytes:
        st.error("Uploaded file is empty.")
        return

    st.image(file_bytes, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze image", type="primary"):
        with st.spinner("Running forensic checks..."):
            try:
                result = get_service().analyze_image(file_bytes, include_heatmap=include_heatmap)
            except ValueError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Failed to analyze image: {exc}")
                return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Label", result.label)
        col2.metric("Final score", f"{result.score:.3f}")
        col3.metric("AI score", f"{result.ai_score:.3f}")
        col4.metric("Edit score", f"{result.edit_score:.3f}")

        st.progress(result.confidence, text=f"Confidence: {result.confidence:.3f}")
        st.write(result.explanation)

        with st.expander("Detection details", expanded=False):
            st.json(result.details)

        if include_heatmap and result.heatmap_base64:
            heatmap_bytes = base64.b64decode(result.heatmap_base64)
            st.image(heatmap_bytes, caption="Forensic heatmap", use_container_width=True)


if __name__ == "__main__":
    main()
