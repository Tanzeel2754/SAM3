# SAM 3 & SAM3 Studio — Overview for Stakeholders

**Document purpose:** Explain what SAM 3 is, what our internal web application does, and why it matters for the organization.  
**Audience:** Management and non-technical stakeholders  
**Last updated:** June 2026

---

## 1. Executive Summary

**SAM 3** (Segment Anything Model 3) is a state-of-the-art AI system from Meta that can find, outline, and track objects in photos and videos using **plain-language instructions** (e.g. *"person in red"*, *"car"*, *"safety helmet"*) instead of training a custom model for every object type.

We have deployed **SAM3 Studio** — a browser-based application that wraps this technology in a simple UI. Users upload an image or video, type what they want to find, and receive visual overlays (masks and bounding boxes) in seconds. No coding is required for day-to-day use.

| Item | Summary |
|------|---------|
| **Technology** | Meta SAM 3 (images) + SAM 3.1 Object Multiplex (video) |
| **Our deliverable** | SAM3 Studio web app (`webapp/`) |
| **How users access it** | Web browser → `http://<server-ip>:7860` |
| **Hardware** | NVIDIA GPU strongly recommended (~12–24 GB VRAM per active model) |
| **Licensing** | SAM 3 checkpoints require Hugging Face access approval from Meta |

---

## 2. What Problem Does SAM 3 Solve?

Traditional computer vision often works like this:

1. Collect thousands of labeled images for each object class  
2. Train or fine-tune a dedicated model  
3. Redeploy when requirements change  

That cycle is slow, expensive, and brittle when new object types or phrasing appear.

SAM 3 flips the model: **one foundation model** understands a huge range of visual concepts. The user describes *what* to segment; the model generalizes without retraining.

### Typical business questions SAM 3 can help answer

- *Where are all the people wearing red in this photo?*  
- *Can we outline every vehicle in this traffic camera frame?*  
- *Can we track a specific object across a video clip?*  
- *Can an analyst refine results with a click or box instead of editing masks by hand?*

SAM 3 does **not** replace domain-specific compliance systems on its own, but it dramatically accelerates **prototyping, annotation, inspection, and demo workflows**.

---

## 3. What Is SAM 3? (Technology Background)

SAM 3 is Meta’s third-generation “Segment Anything” family. Key capabilities:

### Open-vocabulary segmentation

Users prompt with **short text phrases**. The model detects **all instances** matching that concept in an image (e.g. every *"person"* or every *"dog"*).

### Visual prompts

In addition to text, users can guide the model with:

- **Bounding boxes** (include or exclude regions)  
- **Points** (video mode)  
- **Masks** (in research / API contexts)

### Image + video in one research line

- **Images:** detect and mask objects in a single frame  
- **Videos:** detect on a frame, then **propagate** masks across time (tracking)

### Improvements over SAM 2

| Capability | SAM 2 | SAM 3 |
|------------|-------|-------|
| Text-based concept search | Limited | Core feature — open vocabulary |
| “Find all instances of X” | Not primary use case | Primary use case |
| Discrimination between similar prompts | Weaker | Stronger (e.g. “player in white” vs “player in red”) |
| Scale of concepts | Smaller | 270K+ concepts in Meta’s SA-CO benchmark |

### SAM 3.1 (video)

Released March 2026. Adds **Object Multiplex** — a faster way to track many objects in video by processing groups jointly. Our video tab uses this stack for propagation and tracking.

**References**

- [SAM 3 paper & project](https://ai.meta.com/sam3)  
- [GitHub repository](https://github.com/facebookresearch/sam3)  
- [Hugging Face checkpoints](https://huggingface.co/facebook/sam3) (access required)

---

## 4. What We Built: SAM3 Studio

SAM3 Studio is an **internal web application** that exposes SAM 3 capabilities through a polished UI. It is built on the same inference APIs Meta documents in their example notebooks.

### Architecture (high level)

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (React UI)                                         │
│  • Image Segmentation tab                                   │
│  • Video Tracking tab                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / API
┌──────────────────────────▼──────────────────────────────────┐
│  Backend (Python / FastAPI)                                 │
│  • Session management (uploads, prompts, results)           │
│  • GPU model loading & memory management                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  AI Models (on NVIDIA GPU)                                  │
│  • Image: SAM 3 (`facebook/sam3`)                           │
│  • Video: SAM 3.1 Multiplex (`facebook/sam3.1`)             │
└─────────────────────────────────────────────────────────────┘
```

### Two modes

#### Image Segmentation

| Feature | Description |
|---------|-------------|
| Upload | Drag-and-drop or file picker (JPEG, PNG, etc.) |
| Text prompt | e.g. `"person"`, `"car"`, `"person in red on the left"` |
| Segment | Runs model; returns masks + boxes for all matching instances |
| Box prompts | Draw **Include** (keep) or **Exclude** (remove) boxes to refine |
| Confidence slider | Filter weak detections (higher = fewer, more precise results) |
| Clear prompts | Reset and start over on the same image |

**Workflow:** Upload → type prompt → Segment → optionally refine with boxes or confidence.

#### Video Tracking

| Feature | Description |
|---------|-------------|
| Upload | MP4 / WebM (with optional **trim** before processing) |
| Frame navigation | Scrub timeline or play through frames |
| Text prompt | Describe object on a chosen frame |
| Point prompts | Click positive/negative points on objects |
| Propagate | Track segmented objects across the rest of the video |
| Overlay toggle | Show or hide segmentation overlay |

**Workflow:** Upload (trim if needed) → pick frame → prompt → propagate through clip.

### GPU memory management

SAM 3 models are large. On a typical workstation GPU (e.g. RTX 3090, 24 GB):

- Each model mode uses a significant portion of VRAM while loaded  
- **Only one mode’s model is kept on GPU at a time**  
- Switching tabs (Image ↔ Video) **unloads** the other model to free memory  
- Refreshing the page also releases stale GPU allocations  

This design keeps the app usable on single-GPU machines without requiring a server restart between tasks.

---

## 5. How It Works (Non-Technical)

1. **User uploads media** → file is stored in a temporary server session  
2. **User describes the target** in natural language (or clicks/draws on the image/video)  
3. **The model runs on the GPU** → produces a probability map per detected instance  
4. **The UI draws colored overlays** → masks outline the object; boxes show location and confidence score  
5. **User iterates** → adjust confidence, add include/exclude boxes, or try a different prompt  

No manual polygon editing is required for initial results. Analysts get a strong starting point in seconds instead of minutes per frame.

---

## 6. Hardware & Infrastructure Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA CUDA GPU strongly recommended |
| **VRAM** | ~12–24 GB per active model; 24 GB cards are comfortable |
| **CPU / RAM** | Moderate; video frame extraction uses CPU |
| **OS** | Windows / Linux (developed and tested on Windows 10/11) |
| **Python** | 3.12+ with PyTorch 2.7+ and CUDA 12.6+ |
| **Network** | Local (`127.0.0.1`) or LAN (`0.0.0.0`) access |

### Running the server

```bash
# From repository root, with conda env activated:
pip install -r webapp/requirements.txt
cd webapp/frontend && npm install && npm run build && cd ../..
python webapp/run.py --host 0.0.0.0 --port 7860
```

- **Local only:** `http://127.0.0.1:7860`  
- **Network access:** `http://<machine-ip>:7860` (requires firewall allow rule for port 7860)

The header bar shows GPU name and free memory so operators can confirm the system is ready.

---

## 7. Business Use Cases

| Domain | Example use |
|--------|-------------|
| **Security & surveillance** | Find all people or vehicles in a frame; track a subject through a clip |
| **Retail / operations** | Count displays, products, or uniforms matching a description |
| **Sports & events** | Segment players by jersey color or role |
| **Industrial inspection** | Prototype detection of equipment, PPE, or defects described in text |
| **Data labeling** | Accelerate annotation by auto-generating masks for human review |
| **R&D demos** | Show stakeholders interactive vision AI without building a custom model |

SAM 3 is especially valuable when:

- Requirements change frequently  
- Object categories are diverse or described in language  
- You need a **fast proof-of-concept** before investing in a specialized production pipeline  

---

## 8. Limitations & Risks (Important for Decision-Makers)

### Model limitations

- **Not 100% accurate** — scores and masks can be wrong; human review is advised for critical decisions  
- **Ambiguous prompts** fail more often — prefer clear phrases (*"red car"*) over vague ones  
- **Similar objects** can be confused — confidence tuning and exclude boxes help  
- **Video length** — long uploads are processed frame-by-frame; trim clips to what you need  
- **First run is slow** — models load from disk/Hugging Face on first use  

### Operational limitations

- **Single-server, in-memory sessions** — refreshing the browser or restarting the server clears active work  
- **No built-in user authentication** — do not expose to the public internet without security review  
- **GPU-bound throughput** — one heavy job at a time per GPU is realistic  
- **Checkpoint access** — Meta requires Hugging Face approval to download model weights  

### Compliance

- Review Meta’s SAM 3 **license and acceptable use policy** before production deployment  
- Uploaded media may contain PII — treat the server and upload directory according to your data governance rules  

---

## 9. Comparison: SAM3 Studio vs. Alternatives

| Approach | Time to first result | Flexibility | Cost profile |
|----------|---------------------|-------------|--------------|
| **Manual annotation** | Hours per image | Total control | High labor cost |
| **Custom trained detector** | Weeks–months | Fixed classes only | High upfront, low per-class marginal cost |
| **SAM 3 / SAM3 Studio** | Seconds–minutes | Open vocabulary, interactive | GPU hardware + engineering setup; low per-query marginal cost |

SAM3 Studio is best positioned as an **interactive analysis and prototyping tool**, not necessarily a drop-in replacement for a certified production vision system without further validation.

---

## 10. Current Status & What’s Included in This Repository

| Asset | Location | Role |
|-------|----------|------|
| Meta SAM 3 codebase | Repository root | Core AI library |
| SAM3 Studio backend | `webapp/backend/` | API, sessions, inference |
| SAM3 Studio frontend | `webapp/frontend/` | React UI |
| Startup script | `webapp/run.py` | Single command to launch |
| Operator README | `webapp/README.md` | Install & run instructions |

The web app mirrors capabilities from Meta’s official notebooks:

- Image: `examples/sam3_image_interactive.ipynb`  
- Video: `examples/sam3.1_video_predictor_example.ipynb`  

---

## 11. Recommended Next Steps (If We Want to Go Further)

1. **Pilot with real data** — 2–3 use cases with domain experts; measure time saved vs manual review  
2. **Define acceptance criteria** — accuracy thresholds per use case before any production reliance  
3. **Security hardening** — authentication, HTTPS, network segmentation if used beyond a lab LAN  
4. **Integration** — export masks/JSON to existing pipelines (analytics, CMMS, labeling tools)  
5. **Scaling** — dedicated inference server, job queue, or multi-GPU if concurrent users are needed  

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| **Segmentation** | Drawing a precise outline (mask) around an object in an image |
| **Bounding box** | Rectangle around an object |
| **Prompt** | User instruction — text, point, or box — telling the model what to find |
| **Confidence score** | Model’s certainty (0–1); higher threshold = fewer but more reliable detections |
| **Propagation** | Carrying segmentation from one video frame forward through time (tracking) |
| **VRAM** | GPU memory; models must fit here to run at full speed |
| **Open vocabulary** | Ability to understand many object descriptions without retraining |

---

## 13. One-Paragraph Elevator Pitch

> SAM 3 is Meta’s latest AI for finding and outlining objects in images and videos using everyday language. Instead of building a new detector for every object type, we run one powerful model and tell it what to look for. SAM3 Studio puts that capability in a web browser so analysts and stakeholders can upload media, type a description like “person in red,” and immediately see labeled results they can refine with a slider or a mouse. It runs on our GPU server, keeps memory under control by loading one mode at a time, and is ideal for fast demos, labeling assistance, and exploring computer vision use cases before we commit to custom engineering.

---

*For technical setup details, see [webapp/README.md](./README.md). For the upstream SAM 3 research project, see the [repository root README](../README.md).*
