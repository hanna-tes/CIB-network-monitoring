# CIB-network-monitoring Dashboard

A Streamlit-based dashboard to analyze social media data and detect patterns of coordinated sharing, bot activity, and disinformation campaigns.

---

## Features

- Upload CSV or Excel files containing social media post data
- View summary statistics (top users, hashtags, shared URLs)
- Visualize posting spikes over time
- Analyze user behavior (most active users, repeated sharing)
- Detect coordination via:
  - Shared URLs posted in quick succession
  - Hashtag/content overlap
  - Visual and visual–text similarity via CLIP
  - Multilingual search with translation caching
- Explore interactive user interaction networks
- Export visualizations and findings

---

## Input Data Format

Your dataset should include at least the following columns:

- `user_name`
- `timestamp`
- `content` or `text`
- `hashtags`
- `urls`
- `mentions` (optional)
- `retweet_from` (optional)

---

## Getting Started

### 1. Clone the repository

```bash
git clone
