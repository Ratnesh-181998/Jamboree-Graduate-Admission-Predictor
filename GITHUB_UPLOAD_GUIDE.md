# 📤 GitHub Upload Guide

Follow these steps to upload your Jamboree Graduate Admission Predictor project to GitHub.

---

## 🚀 Quick Upload Steps

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click **"New Repository"** (+ icon in top right)
3. Fill in repository details:
   - **Repository name:** `Graduate-Admission-Predictor-Machine-Learning`
   - **Description:** 
     ```
     🎓 ML College Admission Predictor: Estimate IVY League acceptance probability using GRE, TOEFL, CGPA. 4 regression models, 82% accuracy. Interactive Streamlit dashboard with data visualization & predictions.
     ```
   - **Visibility:** Public
   - **DO NOT** initialize with README (we already have one)
4. Click **"Create repository"**

---

### 2. Initialize Git (If Not Already Done)

Open terminal in project directory and run:

```bash
cd C:\Users\rattu\Downloads\Jamboree_Regression_analysis
git init
```

---

### 3. Add Files to Git

```bash
# Add all files
git add .

# Check status
git status
```

---

### 4. Commit Changes

```bash
git commit -m "Initial commit: Graduate Admission Predictor with Streamlit dashboard"
```

---

### 5. Connect to GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/Graduate-Admission-Predictor-Machine-Learning.git
```

---

### 6. Push to GitHub

```bash
# For first push
git branch -M main
git push -u origin main
```

---

## 🏷️ Add Topics/Tags on GitHub

After uploading, go to your repository page and click **"Add topics"**:

```
graduate-admission-prediction
college-admission-calculator
machine-learning-education
ivy-league-predictor
gre-score-analysis
toefl-prediction
linear-regression-model
streamlit-dashboard
python-data-science
educational-analytics
scikit-learn-project
admission-forecasting
```

---

## 📝 Update Repository Settings

### About Section
1. Click the ⚙️ icon next to "About"
2. Add website (if deployed): `https://your-app.streamlit.app`
3. Check ✅ "Releases"
4. Check ✅ "Packages"

### Social Preview
1. Go to Settings → General
2. Scroll to "Social preview"
3. Upload a screenshot of your app (1280x640px recommended)

---

## 🔗 Optional: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set main file: `Jamboree_App.py`
6. Click **"Deploy"**

---

## ✅ Checklist Before Upload

- [x] README.md created
- [x] requirements.txt created
- [x] .gitignore created
- [x] LICENSE created
- [ ] Update README with your personal links
- [ ] Replace placeholder screenshots
- [ ] Test app locally one more time
- [ ] Remove any sensitive data/API keys

---

## 🎯 Repository URL Structure

Your final repository will be at:
```
https://github.com/YOUR_USERNAME/Graduate-Admission-Predictor-Machine-Learning
```

---

## 📊 After Upload - Next Steps

1. **Star your own repo** ⭐
2. **Share on LinkedIn** with project description
3. **Add to portfolio** website
4. **Create a release** (v1.0.0)
5. **Enable GitHub Pages** for documentation

---

## 🆘 Troubleshooting

### If you get "remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/Graduate-Admission-Predictor-Machine-Learning.git
```

### If you need to update after first push:
```bash
git add .
git commit -m "Update: Description of changes"
git push
```

### If files are too large:
- CSV files are auto-downloaded by the app
- PDFs can be added to `.gitignore` if needed

---

## 📧 Need Help?

If you encounter any issues, check:
- [GitHub Docs](https://docs.github.com)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)

---

**Good luck with your upload! 🚀**
