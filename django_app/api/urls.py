from django.urls import path
from .views import EmotionPredictionView

def home(request):
    return HttpResponse("Welcome to the Emotion Prediction API!")

urlpatterns = [
    path('predict/', EmotionPredictionView.as_view(), name='predict'),
]