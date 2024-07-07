from django.http import JsonResponse , HttpResponse
from utils import translater , catalogueLimiter
import json
# from django.views.decorators.csrf import csrf_exempt
import base64

def home(request):
    data = {"status": "Working Good with Status 200"}
    return JsonResponse(data , safe=False)

def translateTo(request):
    qoute = request.GET.get('q')
    source = request.GET.get('src')
    destination = request.GET.get('dest')
    result = translater.translateIn(qoute , source , destination )
    result["org"] = qoute
    return JsonResponse(result ,safe=False, json_dumps_params={'ensure_ascii': False})

def translationList(request):
    data = {
        "result" : translater.getAllTranslationList()
    }
    return JsonResponse(data)

def catalogReader(request):
    data = request.GET.get('data')
    if(data):
        decode = base64.b64decode(data).decode("ascii")
        # payload = json.loads(decode)
        data = catalogueLimiter.catalogueRater(decode)
        return JsonResponse(data , safe=False)
    else:
        return JsonResponse({
            "Message" : "This is the catalog reader api endpoint, put your data according to '[{'price': 576.0, 'rating': 3.0, 'promotion': 1, 'attributes': ['Badminton Set', 'Electronics']}]' this data should be input in form of JSON stringify then further Base64Encoding will done"
        } , safe=False)
        