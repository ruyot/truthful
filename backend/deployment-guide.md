# Deployment Guide

## Backend Deployment (Render)

### 1. Prepare for Render Deployment

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Use these settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
   - **Python Version**: 3.9.18

### 2. Environment Variables on Render

Set these environment variables in your Render dashboard:

```
PORT=10000
PYTHON_VERSION=3.9.18
SENTRY_DSN=your-sentry-dsn-here (optional)
```

### 3. System Dependencies

Render automatically installs:
- FFmpeg (for video processing)
- Python packages from requirements.txt

### 4. CORS Configuration

The backend is configured to accept requests from:
- `*.netlify.app` domains
- `*.netlify.com` domains
- Local development URLs

## Frontend Deployment (Netlify)

### 1. Prepare for Netlify Deployment

1. Build the project: `npm run build`
2. Deploy the `dist` folder to Netlify
3. Or connect your GitHub repository for automatic deployments

### 2. Environment Variables on Netlify

Set these in your Netlify dashboard:

```
VITE_BACKEND_URL=https://your-render-app.onrender.com
VITE_SUPABASE_URL=https://your-project-id.supabase.co (optional)
VITE_SUPABASE_ANON_KEY=your-anon-key (optional)
```

### 3. Build Settings

- **Build command**: `npm run build`
- **Publish directory**: `dist`
- **Node version**: 18

## Testing the Deployment

1. **Backend Health Check**: Visit `https://your-render-app.onrender.com/health`
2. **Frontend**: Visit your Netlify URL
3. **API Integration**: Test video analysis through the frontend

## Monitoring

### Sentry Integration (Optional)

1. Create a Sentry account
2. Get your DSN
3. Add `SENTRY_DSN` environment variable to Render
4. Uncomment Sentry initialization in `main.py`

### Render Monitoring

- View logs in Render dashboard
- Monitor performance and uptime
- Set up alerts for downtime

## Scaling Considerations

### Backend (Render)
- Start with Basic plan ($7/month)
- Upgrade to Standard for higher traffic
- Consider adding Redis for caching

### Frontend (Netlify)
- Free tier supports most use cases
- Pro plan for custom domains and advanced features

## Security

1. **CORS**: Properly configured for production domains
2. **Rate Limiting**: Consider adding rate limiting for API endpoints
3. **Input Validation**: All inputs are validated
4. **Error Handling**: Comprehensive error handling with Sentry

## Performance Optimization

1. **Video Processing**: Limited to 100MB uploads
2. **Frame Sampling**: 3 FPS for balance of speed and accuracy
3. **Cleanup**: Temporary files are automatically cleaned up
4. **Caching**: Consider adding Redis for repeated analyses

## Troubleshooting

### Common Issues:

1. **CORS Errors**: Update CORS origins in `main.py`
2. **Build Failures**: Check Python version and dependencies
3. **Memory Issues**: Optimize video processing or upgrade plan
4. **Timeout Issues**: Increase timeout limits for large videos

### Debug Steps:

1. Check Render logs for backend issues
2. Check browser console for frontend issues
3. Test API endpoints directly
4. Verify environment variables are set correctly