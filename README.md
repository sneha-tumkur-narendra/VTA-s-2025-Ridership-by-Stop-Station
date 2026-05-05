# 🚌 VTA 2025 Ridership by Stop/Station

An end-to-end data analysis and visualization project exploring **Santa Clara Valley Transportation Authority (VTA)** ridership patterns across bus stops and light rail stations in 2025.

> **Course:** DATA 230 - Group 7 | **Year:** 2026

---

## 📌 Project Overview

This project analyzes VTA ridership data at the stop and station level to uncover trends in public transit usage across Santa Clara County. The goal is to help transit planners and the public better understand which stops and routes carry the most riders, how ridership varies by city and route type, and where resources may be most needed.

---

## 📁 Repository Structure

```
VTA-s-2025-Ridership-by-Stop-Station/
│
├── EDA/                              # Exploratory Data Analysis notebooks
│   └── *.ipynb                       # Data cleaning, profiling, and initial visualizations
│
├── Ploty Dash Dashboard/             # Interactive Plotly Dash web application
│   └── vta_dashboard_cpu.ipynb       # Multi-tab dashboard with 9 visuals & 4 dynamic KPIs
│
├── tableau_dashboard/                # Tableau workbook and exports
│   └── *.twb / *.png                 # Static and interactive Tableau dashboards
│
├── DATA230_Group7_Mid-Presentation.pdf   # Midterm presentation slides
└── README.md
```

---

## 📊 Key Features

### 🔍 Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing of VTA ridership records
- Statistical summaries by stop, route, city, and route type
- Distribution analysis and outlier detection
- Correlation analysis across ridership metrics

### 📈 Plotly Dash Dashboard
An interactive multi-tab web application featuring:
- **3 interactive filter components** - filter by city and route type
- **4 Dash callbacks** for dynamic, real-time updates
- **9 visualizations** including:
  - Bubble chart (ridership volume by stop)
  - Balance chart (boardings vs. alightings)
  - Load chart (passenger load by route)
- **4 dynamic KPI cards** updating with filters

### 📉 Tableau Dashboard
- Visual storytelling with geographic and temporal ridership breakdowns
- Station-level heatmaps and bar charts
- Route performance comparisons

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core programming language |
| Pandas | Data manipulation |
| Plotly | Interactive charting |
| Dash | Web dashboard framework |
| Jupyter Lab | Development environment |
| Tableau | Business intelligence dashboards |
| Anaconda | Environment management |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas plotly dash jupyter
```

### Run the Plotly Dash Dashboard

1. Clone the repository:
```bash
git clone https://github.com/sneha-tumkur-narendra/VTA-s-2025-Ridership-by-Stop-Station.git
cd VTA-s-2025-Ridership-by-Stop-Station
```

2. Open the dashboard notebook:
```bash
jupyter lab "Ploty Dash Dashboard/vta_dashboard_cpu.ipynb"
```

3. Run all cells and open your browser to:
```
http://127.0.0.1:8050
```

> **Note:** If port 8050 is already in use, run this in terminal first:
> ```bash
> lsof -ti:8050 | xargs kill -9
> ```

---

## 📂 Data Source

- **Source:** Santa Clara Valley Transportation Authority (VTA)
- **Dataset:** 2025 Ridership by Stop/Station
- **Coverage:** Bus stops and light rail stations across Santa Clara County, CA

---

## 👥 Team

**DATA 230 - Group 7**

| Name |
|------|
| Sneha Tumkur Narendra |
| Sadaf Fatima Syeda |
| Naeem Mannan |

---

## 📄 License

This project is for academic purposes as part of the DATA 230 coursework.

---

## 🙏 Acknowledgements

- Santa Clara Valley Transportation Authority (VTA) for providing open ridership data
- **Prof. Guannan Liu** and **ISA Andrew** for guidance and support throughout the course

## Final presentation link 
- https://datastudio.google.com/reporting/1875b7e8-7b86-494a-9a62-2f4110f9d021/page/tG4wF
