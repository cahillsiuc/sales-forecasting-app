# Admin Guide - AI Sales Forecasting App

## Admin Access

**Admin Credentials:**
- Username: `admin`
- Password: `Admin123!`
- Role: Admin (full access)

**Customer Credentials:**
- Username: `customer`
- Password: `Customer123!`
- Role: Customer (data isolated)

## Admin Features

### 1. Admin Panel
- Access via "🔧 Admin Panel" button in the sidebar (visible only to admin users)
- View all users' uploaded data
- Browse user-specific files and forecasts
- View summary statistics for all users

### 2. Data Isolation
- Customer users can only see their own uploaded data
- Admin users can see all data from all users
- Each user's data is stored in `user_data/{username}/` directory

### 3. User Management

#### Adding New Users

1. **Edit `config.yaml` file:**
```yaml
credentials:
  usernames:
    newusername:
      email: user@example.com
      name: User Full Name
      password: $2b$12$... # Generate using generate_passwords.py
      role: customer  # or 'admin'
```

2. **Generate Password Hash:**
```bash
# Run in virtual environment
python generate_passwords.py
# Copy the generated hash to config.yaml
```

3. **Commit changes to GitHub:**
```bash
git add config.yaml
git commit -m "Add new user"
git push
```

4. **Streamlit Cloud will auto-redeploy** (takes ~2-3 minutes)

#### Changing User Passwords

1. Generate new password hash using `generate_passwords.py`
2. Update the password field in `config.yaml`
3. Commit and push changes
4. Wait for auto-deployment

#### Removing Users

1. Remove user entry from `config.yaml`
2. Optionally delete user's data folder: `user_data/{username}/`
3. Commit and push changes

### 4. Deployment Management

#### Updating the Application

1. **Make changes locally:**
```bash
# Edit files as needed
git add .
git commit -m "Description of changes"
git push
```

2. **Streamlit Cloud auto-deploys:**
   - Monitors GitHub repository
   - Automatically rebuilds on push to main branch
   - Takes 2-5 minutes typically

#### Viewing Deployment Logs

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click "Manage app"
4. View logs in real-time

#### Manual Reboot

If the app is unresponsive:
1. Go to app management page
2. Click "Reboot app"
3. Wait 30-60 seconds

### 5. Managing Secrets

#### In Streamlit Cloud Dashboard:

1. Go to app management page
2. Click "Secrets" in the left sidebar
3. Add/update secrets in TOML format:
```toml
OPENAI_API_KEY = "sk-..."
```
4. Click "Save"
5. App automatically restarts

#### Never commit secrets to GitHub:
- `.env` files are gitignored
- `secrets.toml` is gitignored
- Use `env.template` and `.streamlit/secrets.toml.template` for reference

### 6. Monitoring Usage

#### OpenAI API Usage:
- Monitor at [platform.openai.com/usage](https://platform.openai.com/usage)
- Set usage limits to prevent unexpected charges
- Typical usage: ~$0.002 per AI chat question

#### Streamlit Cloud Usage:
- Free tier: Unlimited public apps
- Check resource usage in dashboard
- Upgrade to paid plan if needed

### 7. Data Backup

#### User Data Location:
- `user_data/{username}/` contains all user uploads
- Files are timestamped: `upload_YYYYMMDD_HHMMSS.csv`

#### Backup Strategy:
1. **Automated (Recommended):**
   - Set up GitHub Actions to backup user_data/ to cloud storage
   - Or use Streamlit Cloud's backup features

2. **Manual:**
```bash
# Download from Streamlit Cloud (if accessible)
# Or pull from GitHub if user_data is tracked
```

3. **Note:** By default, `user_data/` is gitignored for security
   - Only commit if data is non-sensitive
   - Use encrypted backup solution for production

### 8. Troubleshooting

#### Users Can't Login
- Verify credentials in `config.yaml`
- Check password hash is correct
- Ensure latest deployment is live
- Check Streamlit Cloud logs

#### App Won't Start
- Check Streamlit Cloud logs for errors
- Verify `requirements.txt` dependencies
- Check `config.yaml` syntax (valid YAML)
- Verify secrets are configured

#### OpenAI API Not Working
- Verify API key in Streamlit secrets
- Check API key is valid and has credits
- Check usage limits not exceeded
- Review error messages in logs

#### Data Not Saving
- Check `user_data/` directory permissions
- Verify file paths are correct
- Check Streamlit Cloud storage limits
- Review logs for write errors

### 9. Security Best Practices

1. **Change Default Passwords:**
   - Update admin and customer passwords immediately after deployment
   - Use strong passwords (12+ characters, mixed case, numbers, symbols)

2. **Secure API Keys:**
   - Never commit API keys to GitHub
   - Rotate keys periodically
   - Use environment-specific keys

3. **Monitor Access:**
   - Review admin panel regularly
   - Check for suspicious login attempts
   - Monitor API usage for anomalies

4. **Regular Updates:**
   - Keep dependencies updated
   - Monitor security advisories
   - Test updates in staging before production

### 10. Support & Maintenance

#### Regular Tasks:
- [ ] Weekly: Check deployment logs
- [ ] Weekly: Review API usage and costs
- [ ] Monthly: Update dependencies
- [ ] Monthly: Backup user data
- [ ] Quarterly: Rotate API keys
- [ ] Quarterly: Review user access

#### Getting Help:
- Streamlit Community: [discuss.streamlit.io](https://discuss.streamlit.io)
- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- OpenAI Support: [help.openai.com](https://help.openai.com)

## Quick Reference

### File Structure
```
sales-forecasting-demo/
├── app.py                 # Main application
├── config.yaml           # User credentials (SENSITIVE)
├── requirements.txt      # Python dependencies
├── .streamlit/
│   ├── config.toml      # App configuration
│   └── secrets.toml     # API keys (NOT IN GIT)
├── user_data/           # User uploads (NOT IN GIT)
│   ├── admin/
│   └── customer/
├── utils/               # Helper modules
└── sample_data/         # Demo data

```

### Important URLs
- **App URL:** https://your-app.streamlit.app
- **GitHub Repo:** https://github.com/YOUR_USERNAME/YOUR_REPO
- **Streamlit Dashboard:** https://share.streamlit.io
- **OpenAI Dashboard:** https://platform.openai.com

### Support Contacts
- **Technical Issues:** [Your support email]
- **Billing Questions:** [Your billing email]
- **Emergency:** [Your emergency contact]

---

**Last Updated:** October 2025
**Version:** 1.0

