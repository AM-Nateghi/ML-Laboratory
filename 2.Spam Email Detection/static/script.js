// Spam Email Detection Frontend JavaScript
// Handles UI interactions and API communication

// DOM Elements
const emailInput = document.getElementById('emailInput');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const resultSection = document.getElementById('resultSection');
const predictionBadge = document.getElementById('predictionBadge');
const predictionText = document.getElementById('predictionText');
const resultMessage = document.getElementById('resultMessage');
const confidenceFill = document.getElementById('confidenceFill');
const confidencePercentage = document.getElementById('confidencePercentage');
const modelName = document.getElementById('modelName');
const resultIcon = document.querySelector('.result-icon');

// Sample emails
const sampleEmails = {
    spam: `From: winner@lottery-claims.org
Subject: 🎉 CONGRATULATIONS! You've Won $1,000,000

Dear Lucky Winner,

CONGRATULATIONS! You have been selected as the GRAND PRIZE WINNER of our International Email Lottery!

Your winning numbers are: 07-14-21-28-35-42
Prize Amount: $1,000,000 USD

To claim your prize immediately, please provide:
- Full Name
- Address  
- Phone Number
- Bank Account Details

Click here NOW to claim your winnings before this offer expires!

Congratulations again!
Lottery Commission`,

    legitimate: `From: support@legitcompany.com
Subject: Regarding Your Recent Inquiry

Dear Customer,

Thank you for reaching out to us regarding your recent inquiry about our services. We have reviewed your request and wanted to provide you with a comprehensive response.

Based on your questions, I've attached our product catalog and pricing information. Our team is committed to providing you with the best possible service and solutions for your needs.

If you have any additional questions or need further clarification, please don't hesitate to contact our support team at support@legitcompany.com or call us at (555) 123-4567.

We appreciate your interest in our company and look forward to working with you.

Best regards,
Customer Service Team
LegitCompany Inc.`
};

// Character counter
emailInput.addEventListener('input', function() {
    const length = this.value.length;
    charCount.textContent = length;
    
    // Change color based on length
    if (length > 9500) {
        charCount.style.color = '#e53e3e';
    } else if (length > 8000) {
        charCount.style.color = '#dd6b20';
    } else {
        charCount.style.color = '#718096';
    }
    
    // Hide result when text changes
    hideResult();
});

// Load sample emails
function loadSpamEmail() {
    emailInput.value = sampleEmails.spam;
    emailInput.dispatchEvent(new Event('input'));
    hideResult();
}

function loadLegitimateEmail() {
    emailInput.value = sampleEmails.legitimate;
    emailInput.dispatchEvent(new Event('input'));
    hideResult();
}

function clearEmail() {
    emailInput.value = '';
    emailInput.dispatchEvent(new Event('input'));
    hideResult();
}

// Validation function
function validateEmail(email) {
    const errors = [];
    
    if (!email.trim()) {
        errors.push('لطفاً متن ایمیل را وارد کنید');
    } else if (email.trim().length < 10) {
        errors.push('متن ایمیل باید حداقل ۱۰ کاراکتر باشد');
    } else if (email.length > 10000) {
        errors.push('متن ایمیل نمی‌تواند بیش از ۱۰۰۰۰ کاراکتر باشد');
    }
    
    return errors;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    hideResult();
}

// Hide error message
function hideError() {
    errorMessage.classList.add('hidden');
}

// Show result
function showResult(prediction) {
    hideError();
    
    // Update prediction badge
    predictionText.textContent = prediction.prediction;
    predictionBadge.className = `prediction-badge ${prediction.is_spam ? 'spam' : 'legitimate'}`;
    
    // Update result message
    resultMessage.textContent = prediction.message;
    
    // Update confidence visualization
    const confidencePercent = Math.round(prediction.confidence * 100);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidencePercentage.textContent = `${confidencePercent}%`;
    
    // Update confidence bar color based on result
    if (prediction.is_spam) {
        confidenceFill.style.background = 'linear-gradient(90deg, #ff6b6b, #e53e3e)';
        resultIcon.textContent = '⚠️';
    } else {
        confidenceFill.style.background = 'linear-gradient(90deg, #51cf66, #40c057)';
        resultIcon.textContent = '✅';
    }
    
    // Update model name
    modelName.textContent = prediction.model_name;
    
    // Show result section with animation
    resultSection.classList.remove('hidden');
    resultSection.style.opacity = '0';
    resultSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultSection.style.transition = 'all 0.5s ease';
        resultSection.style.opacity = '1';
        resultSection.style.transform = 'translateY(0)';
    }, 100);
}

// Hide result
function hideResult() {
    resultSection.classList.add('hidden');
}

// Set loading state
function setLoading(isLoading) {
    if (isLoading) {
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('loading');
        analyzeBtn.textContent = 'در حال تحلیل...';
        loadingSpinner.style.display = 'block';
    } else {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('loading');
        analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span>تشخیص اسپم<div class="loading-spinner" id="loadingSpinner"></div>';
        loadingSpinner.style.display = 'none';
    }
}

// Main analyze function
async function analyzeEmail() {
    const email = emailInput.value;
    
    // Validate input
    const errors = validateEmail(email);
    if (errors.length > 0) {
        showError(errors[0]);
        return;
    }
    
    setLoading(true);
    hideError();
    hideResult();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email_content: email
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `خطای سرور: ${response.status}`);
        }
        
        const prediction = await response.json();
        showResult(prediction);
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            showError('خطا در برقراری ارتباط با سرور. لطفاً اتصال اینترنت خود را بررسی کنید.');
        } else {
            showError(`خطا در تحلیل ایمیل: ${error.message}`);
        }
    } finally {
        setLoading(false);
    }
}

// Keyboard shortcuts
emailInput.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeEmail();
    }
});

// Enter key on analyze button
analyzeBtn.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        analyzeEmail();
    }
});

// Focus management
document.addEventListener('DOMContentLoaded', function() {
    // Focus on email input when page loads
    emailInput.focus();
    
    // Initialize character counter
    emailInput.dispatchEvent(new Event('input'));
    
    // Check if model is loaded
    checkModelStatus();
});

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (!health.model_loaded) {
            showError('مدل هنوز بارگذاری نشده است. لطفاً چند لحظه صبر کنید...');
            
            // Retry after 3 seconds
            setTimeout(checkModelStatus, 3000);
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Accessibility improvements
function makeAccessible() {
    // Add ARIA labels
    emailInput.setAttribute('aria-label', 'متن ایمیل برای تحلیل');
    analyzeBtn.setAttribute('aria-label', 'تحلیل ایمیل برای تشخیص اسپم');
    
    // Add keyboard navigation for sample buttons
    const sampleButtons = document.querySelectorAll('.sample-btn');
    sampleButtons.forEach(btn => {
        btn.setAttribute('tabindex', '0');
        btn.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });
}

// Initialize accessibility
makeAccessible();

// Smooth scrolling to result
function scrollToResult() {
    if (!resultSection.classList.contains('hidden')) {
        resultSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }
}

// Auto-scroll to result when shown
const resultObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.attributeName === 'class') {
            if (!resultSection.classList.contains('hidden')) {
                setTimeout(scrollToResult, 600); // Wait for animation
            }
        }
    });
});

resultObserver.observe(resultSection, { attributes: true });

// Error handling for network issues
window.addEventListener('online', function() {
    hideError();
});

window.addEventListener('offline', function() {
    showError('اتصال اینترنت قطع شده است. لطفاً اتصال خود را بررسی کنید.');
});

// Prevent form submission on Enter (let users add line breaks)
emailInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey && !(e.ctrlKey || e.metaKey)) {
        // Allow normal Enter for line breaks
        // Only Ctrl/Cmd + Enter triggers analysis
    }
});

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateEmail,
        loadSpamEmail,
        loadLegitimateEmail,
        clearEmail,
        analyzeEmail
    };
}