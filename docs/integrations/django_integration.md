# Django Integration Guide

This guide demonstrates how to integrate Sifaka with Django applications for text analysis, content moderation, and validation.

## Setup

First, install Sifaka and Django:

```bash
pip install django
pip install sifaka
```

## Basic Django Integration

### Project Configuration

Add Sifaka configuration to your Django settings:

```python
# settings.py

INSTALLED_APPS = [
    # ... other apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Optional custom app for Sifaka
    'sifaka_django',
]

# Sifaka configuration
SIFAKA_CONFIG = {
    'cache_size': 1000,
    'default_models': {
        'toxicity': 'original',
        'sentiment': 'vader'
    },
    'api_keys': {
        'openai': 'your-openai-key',  # Optional
        'anthropic': 'your-anthropic-key',  # Optional
    }
}
```

### Creating a Simple Sifaka Django App

Create a new Django app for Sifaka integration:

```bash
python manage.py startapp sifaka_django
```

### Initialize Sifaka in the App

```python
# sifaka_django/apps.py
from django.apps import AppConfig
from django.conf import settings


class SifakaDjangoConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sifaka_django'

    def ready(self):
        # Initialize Sifaka classifiers and rules on app startup
        from sifaka.classifiers import (
            create_toxicity_classifier,
            create_sentiment_classifier
        )
        from sifaka.rules.content import (
            create_prohibited_content_rule,
            create_sentiment_rule
        )

        # Load configuration from settings
        config = getattr(settings, 'SIFAKA_CONFIG', {})
        cache_size = config.get('cache_size', 100)

        # Initialize classifiers
        self.toxicity_classifier = create_toxicity_classifier(
            cache_size=cache_size
        )

        self.sentiment_classifier = create_sentiment_classifier(
            cache_size=cache_size
        )

        # Initialize rules
        self.toxicity_rule = create_prohibited_content_rule(
            classifier=self.toxicity_classifier,
            valid_labels=["non_toxic"],
        )

        self.sentiment_rule = create_sentiment_rule(
            classifier=self.sentiment_classifier,
            valid_labels=["positive", "neutral"],
        )
```

## Content Moderation Middleware

Create middleware to automatically moderate content in requests:

```python
# sifaka_django/middleware.py
from django.http import HttpResponseBadRequest
from django.apps import apps
import json


class ContentModerationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # Get the initialized classifiers from the app config
        self.app_config = apps.get_app_config('sifaka_django')

    def __call__(self, request):
        # Only check POST requests with content
        if request.method == 'POST' and request.content_type == 'application/json':
            try:
                body = json.loads(request.body)

                # Check if there's content to moderate
                if 'content' in body:
                    content = body['content']

                    # Validate content using the toxicity rule
                    result = self.app_config.toxicity_rule.validate(content)

                    if not result.passed:
                        return HttpResponseBadRequest(
                            json.dumps({
                                'error': 'Content moderation failed',
                                'reason': result.message,
                                'metadata': result.metadata
                            }),
                            content_type='application/json'
                        )
            except json.JSONDecodeError:
                pass  # Not JSON data, skip validation

        # Continue processing the request
        return self.get_response(request)
```

Add the middleware to your settings:

```python
# settings.py
MIDDLEWARE = [
    # ... other middleware
    'sifaka_django.middleware.ContentModerationMiddleware',
]
```

## Creating API Views for Text Analysis

```python
# sifaka_django/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.apps import apps
import json


@csrf_exempt
def analyze_text(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if 'text' not in data:
        return JsonResponse({'error': 'No text provided'}, status=400)

    text = data['text']

    # Get app config with initialized classifiers
    app_config = apps.get_app_config('sifaka_django')

    # Analyze sentiment
    sentiment_result = app_config.sentiment_classifier.classify(text)

    # Analyze toxicity
    toxicity_result = app_config.toxicity_classifier.classify(text)

    # Combine results
    result = {
        'sentiment': {
            'label': sentiment_result.label,
            'confidence': sentiment_result.confidence,
            'compound_score': sentiment_result.metadata.get('compound_score', 0),
        },
        'toxicity': {
            'label': toxicity_result.label,
            'confidence': toxicity_result.confidence,
        },
        'text': text,
    }

    return JsonResponse(result)


@csrf_exempt
def validate_content(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if 'content' not in data:
        return JsonResponse({'error': 'No content provided'}, status=400)

    content = data['content']
    app_config = apps.get_app_config('sifaka_django')

    # Validate using toxicity rule
    toxicity_result = app_config.toxicity_rule.validate(content)

    # Validate using sentiment rule
    sentiment_result = app_config.sentiment_rule.validate(content)

    # Combine results
    result = {
        'passed': toxicity_result.passed and sentiment_result.passed,
        'toxicity': {
            'passed': toxicity_result.passed,
            'message': toxicity_result.message,
        },
        'sentiment': {
            'passed': sentiment_result.passed,
            'message': sentiment_result.message,
        },
        'content': content,
    }

    return JsonResponse(result)
```

Add URL patterns for your views:

```python
# sifaka_django/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/analyze/', views.analyze_text, name='analyze_text'),
    path('api/validate/', views.validate_content, name='validate_content'),
]
```

Include the app URLs in your main URLs file:

```python
# project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('sifaka_django.urls')),
]
```

## Django Models with Content Validation

```python
# sifaka_django/models.py
from django.db import models
from django.apps import apps
from django.core.exceptions import ValidationError


class ModeratedContent(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def clean(self):
        # Get app config with initialized rules
        app_config = apps.get_app_config('sifaka_django')

        # Validate content
        result = app_config.toxicity_rule.validate(self.content)

        if not result.passed:
            raise ValidationError({
                'content': f"Content moderation failed: {result.message}"
            })

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)


class Comment(models.Model):
    content = models.ForeignKey(ModeratedContent, on_delete=models.CASCADE, related_name='comments')
    text = models.TextField()
    author = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    sentiment = models.CharField(max_length=20, null=True, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)

    def clean(self):
        # Get app config with initialized rules
        app_config = apps.get_app_config('sifaka_django')

        # Validate content
        toxicity_result = app_config.toxicity_rule.validate(self.text)

        if not toxicity_result.passed:
            raise ValidationError({
                'text': f"Comment moderation failed: {toxicity_result.message}"
            })

        # Analyze sentiment
        sentiment_result = app_config.sentiment_classifier.classify(self.text)
        self.sentiment = sentiment_result.label
        self.sentiment_score = sentiment_result.metadata.get('compound_score', 0)

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)
```

## Forms with Content Validation

```python
# sifaka_django/forms.py
from django import forms
from django.apps import apps
from .models import ModeratedContent, Comment


class ContentForm(forms.ModelForm):
    class Meta:
        model = ModeratedContent
        fields = ['title', 'content', 'author']

    def clean_content(self):
        content = self.cleaned_data.get('content')

        # Get app config with initialized rules
        app_config = apps.get_app_config('sifaka_django')

        # Validate content
        result = app_config.toxicity_rule.validate(content)

        if not result.passed:
            raise forms.ValidationError(
                f"Content moderation failed: {result.message}"
            )

        return content


class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['text', 'author']

    def clean_text(self):
        text = self.cleaned_data.get('text')

        # Get app config with initialized rules
        app_config = apps.get_app_config('sifaka_django')

        # Validate text
        result = app_config.toxicity_rule.validate(text)

        if not result.passed:
            raise forms.ValidationError(
                f"Comment moderation failed: {result.message}"
            )

        return text
```

## Advanced: Background Processing with Celery

For more intensive processing, you might want to use Celery:

```python
# sifaka_django/tasks.py
from celery import shared_task
from django.apps import apps
from .models import ModeratedContent, Comment


@shared_task
def analyze_content(content_id):
    try:
        content = ModeratedContent.objects.get(id=content_id)

        # Get app config with initialized classifiers
        app_config = apps.get_app_config('sifaka_django')

        # Analyze sentiment
        sentiment_result = app_config.sentiment_classifier.classify(content.content)

        # You could store the analysis results in another model
        # or update the content with metadata
        content.sentiment = sentiment_result.label
        content.sentiment_score = sentiment_result.metadata.get('compound_score', 0)
        content.save(update_fields=['sentiment', 'sentiment_score'])

        return {
            'content_id': content_id,
            'sentiment': sentiment_result.label,
            'score': sentiment_result.metadata.get('compound_score', 0),
        }
    except ModeratedContent.DoesNotExist:
        return {'error': 'Content not found'}


@shared_task
def batch_analyze_comments():
    # Get all unanalyzed comments
    comments = Comment.objects.filter(sentiment__isnull=True).order_by('-created_at')[:100]

    if not comments:
        return {'message': 'No unanalyzed comments found'}

    # Get app config with initialized classifiers
    app_config = apps.get_app_config('sifaka_django')

    results = []
    for comment in comments:
        # Analyze sentiment
        sentiment_result = app_config.sentiment_classifier.classify(comment.text)

        # Update comment
        comment.sentiment = sentiment_result.label
        comment.sentiment_score = sentiment_result.metadata.get('compound_score', 0)

        results.append({
            'comment_id': comment.id,
            'sentiment': sentiment_result.label,
            'score': sentiment_result.metadata.get('compound_score', 0),
        })

    # Bulk update
    Comment.objects.bulk_update(comments, ['sentiment', 'sentiment_score'])

    return {
        'analyzed_count': len(results),
        'results': results
    }
```

## Django Admin Extensions

```python
# sifaka_django/admin.py
from django.contrib import admin
from django.utils.html import format_html
from .models import ModeratedContent, Comment


class CommentInline(admin.TabularInline):
    model = Comment
    extra = 0
    readonly_fields = ['sentiment', 'sentiment_score', 'sentiment_color']

    def sentiment_color(self, obj):
        if obj.sentiment == 'positive':
            color = 'green'
        elif obj.sentiment == 'negative':
            color = 'red'
        else:
            color = 'gray'

        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            obj.sentiment
        )

    sentiment_color.short_description = 'Sentiment'


@admin.register(ModeratedContent)
class ModeratedContentAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'created_at']
    search_fields = ['title', 'content', 'author']
    inlines = [CommentInline]
    actions = ['analyze_content']

    def analyze_content(self, request, queryset):
        from .tasks import analyze_content
        for content in queryset:
            analyze_content.delay(content.id)

        self.message_user(request, f"Scheduled {queryset.count()} content items for analysis.")

    analyze_content.short_description = "Analyze selected content"


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['text_preview', 'author', 'sentiment', 'sentiment_score', 'created_at']
    list_filter = ['sentiment', 'created_at']
    search_fields = ['text', 'author']

    def text_preview(self, obj):
        return obj.text[:50] + '...' if len(obj.text) > 50 else obj.text

    text_preview.short_description = 'Comment'
```

## Class-Based Views Example

```python
# sifaka_django/views.py (additional views)
from django.views.generic import CreateView, ListView, DetailView
from django.contrib import messages
from django.urls import reverse_lazy
from .models import ModeratedContent, Comment
from .forms import ContentForm, CommentForm


class ContentCreateView(CreateView):
    model = ModeratedContent
    form_class = ContentForm
    template_name = 'sifaka_django/content_form.html'
    success_url = reverse_lazy('content_list')

    def form_valid(self, form):
        messages.success(self.request, "Content created successfully!")
        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(self.request, "Content validation failed.")
        return super().form_invalid(form)


class ContentListView(ListView):
    model = ModeratedContent
    template_name = 'sifaka_django/content_list.html'
    context_object_name = 'contents'
    paginate_by = 10
    ordering = ['-created_at']


class ContentDetailView(DetailView):
    model = ModeratedContent
    template_name = 'sifaka_django/content_detail.html'
    context_object_name = 'content'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comment_form'] = CommentForm()
        return context

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = CommentForm(request.POST)

        if form.is_valid():
            comment = form.save(commit=False)
            comment.content = self.object
            comment.save()
            messages.success(request, "Comment added successfully!")
            return self.get(request, *args, **kwargs)

        context = self.get_context_data(object=self.object)
        context['comment_form'] = form
        return self.render_to_response(context)
```

## Summary

This integration guide demonstrates how to use Sifaka with Django for:

1. Content moderation using middleware
2. API endpoints for text analysis
3. Model validation for content moderation
4. Form validation for user input
5. Background processing with Celery
6. Admin interface customization
7. Class-based views for content creation and display

These patterns can be adapted to fit your specific requirements, whether you're building a content platform, social media site, or any application that needs text analysis and moderation capabilities.

For more advanced configurations and performance optimization, see the additional guide on [Optimizing Sifaka in Production Environments](/docs/optimization.md).