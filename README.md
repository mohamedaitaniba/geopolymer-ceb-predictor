# 🧱 Geopolymer-Stabilized Soil CEB Predictor

> **Internship Project**  
> **Host Institution:** Green Energy Park (GEP), Morocco  
> **Focus Area:** Sustainable Materials, Circular Economy, and Machine Learning in Civil Engineering  

This repository contains the machine learning pipelines and interactive web interface developed during my internship at the **Green Energy Park (GEP)**. The objective of this project is to accelerate the formulation design of sustainable construction materials—specifically **Geopolymer-Stabilized Compressed Earth Blocks (CEBs)**—by using predictive artificial intelligence to bypass time-consuming laboratory trial-and-error.

---

## 📋 Project Overview

Compressed Earth Blocks (CEBs) stabilized with alkali-activated binders (geopolymers) present a highly ecological, low-carbon alternative to traditional cement-stabilized bricks. However, designing these mixes is complex due to the highly variable nature of soil composition (clay, silt, sand fractions) and the sensitive chemical balances of the alkali activators.

To solve this, I compiled a database of mix designs from academic literature and built an **ensemble-based Machine Learning application** that models and predicts two key performance indicators:
1. **Unconfined Compressive Strength (UCS)** — ensuring structural integrity.
2. **24-hour Water Absorption** — evaluating durability and weather resistance.

---

## 🛠️ System Architecture & ML Approach

Real-world materials data is often incomplete. To address this, the pipeline is engineered with a focus on data resiliency:
* **Dual-Pipeline Architecture:** Rather than using a single multi-output model (which would force the deletion of any row missing *either* target metric), I trained **two independent Random Forest Regressors**. This preserved **201 training instances for UCS** and **88 training instances for Water Absorption** from the master database.
* **K-Nearest Neighbors (KNN) Imputation:** Missing input parameters (such as the chemical silica modulus, which is often unreported in non-alkali mixes) are dynamically filled using a `KNNImputer`. It estimates missing data based on mathematically similar mix compositions.
* **Gradio Web Interface:** Designed an intuitive dashboard featuring sliders and numeric controls, allowing researchers to tweak parameters and observe immediate property simulations.

---

## 📊 Features & Target Parameters

### Model Inputs (Features)
* **Soil Particle Size Distribution:** Clay Content (%), Silt Content (%), and Sand Content (%).
* **Activator Chemistry:** Silica Modulus ($SiO_2/Na_2O$).
* **Precursor Configuration:** Precursor $SiO_2/Al_2O_3$ Ratio (e.g., fly ash, metakaolin, or volcanic ash).

### Model Outputs (Targets)
* **Predicted UCS (MPa)** 
* **Predicted 24-hour Water Absorption (%)**

---

## 🚀 Local Installation & Setup

To run this predictor locally on your machine, follow these steps:

1. **Clone this repository** (or download and extract the ZIP file):
   ```bash
   git clone [https://github.com/YOUR_USERNAME/geopolymer-ceb-predictor.git](https://github.com/YOUR_USERNAME/geopolymer-ceb-predictor.git)
   cd geopolymer-ceb-predictor
