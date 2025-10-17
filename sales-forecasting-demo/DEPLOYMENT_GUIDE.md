# Deployment Guide - AI Sales Forecasting App

## Overview
This guide will help you deploy your sales forecasting app to Streamlit Cloud with authentication and data isolation.

## Prerequisites
- GitHub account
- OpenAI API key (get from [platform.openai.com](https://platform.openai.com/api-keys))
- Git installed on your computer

## Step 1: Prepare Your Environment

### 1.1 Install Dependencies
```bash
cd sales-forecasting-demo
.\venv\Scripts\Activate.ps1  # Windows
# OR
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 1.2 Configure OpenAI API Key (Local Testing)
Create a `.env` file in the `sales-forecasting-demo/` directory:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

**Never commit this file to GitHub!** (It's already in .gitignore)

### 1.3 Test Locally
```bash
streamlit run app.py
```

Test with both accounts:
- **Admin:** username `admin`, password `Admin123!`
- **Customer:** username `customer`, password `Customer123!`

## Step 2: Initialize Git Repository

### 2.1 Initialize Git (if not already done)
```bash
git init
```

### 2.2 Add Files to Git
```bash
git add .
git commit -m "Initial commit with authentication"
```

## Step 3: Create GitHub Repository

### 3.1 On GitHub.com:
1. Go to [github.com](https://github.com)
2. Click the "+" icon in top right
3. Select "New repository"
4. Name it (e.g., `sales-forecasting-app`)
5. Choose "Private" (recommended for production)
6. DO NOT initialize with README (you already have files)
7. Click "Create repository"

### 3.2 Connect Local to GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/sales-forecasting-app.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud

### 4.1 Sign Up for Streamlit Cloud:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up" or "Continue with GitHub"
3. Authorize Streamlit to access your GitHub account

### 4.2 Deploy Your App:
1. Click "New app" button
2. Select your repository: `YOUR_USERNAME/sales-forecasting-app`
3. Select branch: `main`
4. Set main file path: `app.py`
5. Click "Deploy!"

### 4.3 Wait for Deployment:
- Initial deployment takes 2-5 minutes
- Watch the logs for any errors
- The app URL will be: `https://YOUR-APP-NAME.streamlit.app`

## Step 5: Configure Secrets in Streamlit Cloud

### 5.1 Add OpenAI API Key:
1. Click on your app in the Streamlit dashboard
2. Click "⚙️ Settings" or "Manage app"
3. Click "Secrets" in the left sidebar
4. Add your secrets in TOML format:
```toml
OPENAI_API_KEY = "sk-your-actual-openai-key-here"
```
5. Click "Save"
6. The app will automatically restart

## Step 6: Test Your Deployment

### 6.1 Test Customer Login:
1. Go to your app URL
2. Login with:
   - Username: `customer`
   - Password: `Customer123!`
3. Upload a CSV file
4. Generate a forecast
5. Test AI chat features
6. Verify you can only see your own data

### 6.2 Test Admin Login:
1. Logout (click "Logout" in sidebar)
2. Login with:
   - Username: `admin`
   - Password: `Admin123!`
3. Click "🔧 Admin Panel" in sidebar
4. Verify you can see customer's uploaded data
5. Upload your own file
6. Verify data isolation works

## Step 7: Share with Your Customer

### 7.1 Provide Access Information:
Send your customer:
```
🎉 Your AI Sales Forecasting App is Ready!

App URL: https://your-app-name.streamlit.app

Login Credentials:
- Username: customer
- Password: Customer123!

Features:
✅ Upload your sales data (CSV or Excel)
✅ Generate forecasts with 3 AI models
✅ Get AI-powered business insights
✅ Download forecast results

Your data is secure and isolated - only you can see it!

Need help? Reply to this email.
```

### 7.2 First-Time User Instructions:
Create a quick start guide for your customer (see CUSTOMER_QUICK_START.md)

## Step 8: Post-Deployment Tasks

### 8.1 Change Default Passwords (IMPORTANT!):
1. Edit `config.yaml` locally
2. Run `python generate_passwords.py` to create new password hashes
3. Update passwords in config.yaml
4. Commit and push:
```bash
git add config.yaml
git commit -m "Update default passwords"
git push
```
5. Wait for auto-redeployment

### 8.2 Set Up Monitoring:
- Monitor OpenAI API usage: [platform.openai.com/usage](https://platform.openai.com/usage)
- Set spending limits on OpenAI dashboard
- Bookmark Streamlit app management page

### 8.3 Create Backup Plan:
- Document how to backup user data
- Set up automated backups (optional)
- Keep local copy of config.yaml securely

## Troubleshooting

### App Won't Start
**Problem:** Red error message or app crashes

**Solutions:**
1. Check Streamlit Cloud logs (in app management page)
2. Verify `requirements.txt` has all dependencies
3. Check `config.yaml` is valid YAML (use yamllint.com)
4. Ensure secrets are configured correctly
5. Try manual reboot from dashboard

### Authentication Not Working
**Problem:** Can't login with credentials

**Solutions:**
1. Verify passwords in `config.yaml` are correct hashes
2. Check cookie settings aren't blocking
3. Try incognito/private browsing window
4. Verify config.yaml was pushed to GitHub
5. Check deployment logs for authentication errors

### OpenAI Features Not Working
**Problem:** AI chat returns errors

**Solutions:**
1. Verify `OPENAI_API_KEY` is in Streamlit secrets
2. Check API key is valid at platform.openai.com
3. Verify you have API credits available
4. Check usage limits haven't been exceeded
5. Review app logs for specific error messages

### Files Not Uploading
**Problem:** File upload fails or data doesn't save

**Solutions:**
1. Check file size (must be under 200MB)
2. Verify CSV/Excel format is valid
3. Check Streamlit Cloud storage limits
4. Try smaller file or different format
5. Review logs for specific errors

### App is Slow
**Problem:** App takes long to respond

**Solutions:**
1. Check Streamlit Cloud resource usage
2. Reduce forecast periods
3. Use smaller data sets for testing
4. Consider upgrading Streamlit Cloud plan
5. Optimize heavy computations

## Updating Your App

### Making Changes:
```bash
# Edit files locally
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy (2-5 minutes).

### Rolling Back:
If something breaks:
```bash
git revert HEAD
git push
```

Or use GitHub to revert to a specific commit.

## Cost Breakdown

### Free Tier:
- **Streamlit Cloud:** Free for public apps
- **OpenAI API:** Pay-as-you-go
  - Typical: $0.002 per chat question
  - Estimate: $5-20/month for moderate use

### Paid Options:
- **Streamlit Cloud Pro:** $20/user/month (private apps, more resources)
- **OpenAI Usage:** Monitor and set spending limits

## Security Checklist

- [ ] Changed default passwords
- [ ] API keys in secrets (not in code)
- [ ] Repository is private (for production)
- [ ] .env and secrets.toml in .gitignore
- [ ] Monitoring set up
- [ ] Spending limits configured
- [ ] Backup strategy documented
- [ ] Customer notified of credentials

## Next Steps

1. **Monitor for 1 week:**
   - Check logs daily
   - Monitor API usage
   - Get customer feedback

2. **Gather Feedback:**
   - What features are most used?
   - What's confusing?
   - What's missing?

3. **Iterate:**
   - Add requested features
   - Fix bugs
   - Improve performance

4. **Scale:**
   - Add more users as needed
   - Upgrade plans if necessary
   - Consider custom deployment

## Support Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **OpenAI Docs:** [platform.openai.com/docs](https://platform.openai.com/docs)
- **This Project's README:** See README.md

## Need Help?

If you run into issues:
1. Check the troubleshooting section above
2. Review Streamlit Cloud logs
3. Search Streamlit community forums
4. Post detailed questions with error messages

---

**🎉 Congratulations on deploying your app!**

Your sales forecasting tool is now live and ready for customer testing.

**App URL:** https://your-app-name.streamlit.app

Monitor usage, gather feedback, and iterate based on customer needs.

Good luck! 🚀

