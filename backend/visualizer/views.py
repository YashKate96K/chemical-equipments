from django.db import transaction
from django.http import HttpResponse
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.authentication import TokenAuthentication
from rest_framework.response import Response
from rest_framework import status
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

from .models import Dataset
from .serializers import DatasetSerializer, UploadResponseSerializer
from .utils import (
    parse_and_validate,
    CSVValidationError,
    parse_rows,
    compute_quality,
    compute_correlations,
    compute_variance_skewness,
    kmeans_clusters,
)
from .pdf import build_pdf


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
@transaction.atomic
def upload_csv(request):
    file = request.FILES.get('file')
    if not file:
        return Response({'detail': 'file is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Read the uploaded file once and reuse the bytes for validation and storage
    try:
        file_bytes = file.read()
        summary, preview_csv = parse_and_validate(file_bytes, getattr(file, 'name', ''))
    except CSVValidationError as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    dataset = Dataset.objects.create(
        filename=file.name,
        file=ContentFile(file_bytes, name=file.name),
        summary_json=summary,
        preview_csv=preview_csv,
    )

    ids_to_keep = list(Dataset.objects.order_by('-created_at').values_list('id', flat=True)[:5])
    Dataset.objects.exclude(id__in=ids_to_keep).delete()

    resp = {
        'id': dataset.id,
        'filename': dataset.filename,
        'summary_json': dataset.summary_json,
        'preview_csv': dataset.preview_csv,
    }
    return Response(resp, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@authentication_classes([])  # disable auth for signup; allow anonymous registration
@permission_classes([AllowAny])
def register_user(request):
    username = request.data.get('username')
    password = request.data.get('password')
    if not username or not password:
        return Response({'detail': 'username and password are required'}, status=status.HTTP_400_BAD_REQUEST)
    if User.objects.filter(username=username).exists():
        return Response({'detail': 'username already exists'}, status=status.HTTP_400_BAD_REQUEST)
    user = User.objects.create_user(username=username, password=password)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({'token': token.key, 'username': user.username}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def list_datasets(request):
    qs = Dataset.objects.order_by('-created_at')[:5]
    data = DatasetSerializer(qs, many=True).data
    return Response(data)


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def dataset_report(request, pk: int):
    try:
        dataset = Dataset.objects.get(pk=pk)
    except Dataset.DoesNotExist:
        return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

    pdf_bytes = build_pdf(dataset)
    resp = HttpResponse(pdf_bytes, content_type='application/pdf')
    resp['Content-Disposition'] = f'attachment; filename="dataset_{pk}_report.pdf"'
    return resp


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def dataset_health(request, pk: int):
    """Return rows, anomaly indices, and KPIs for the dashboard UI."""
    try:
        dataset = Dataset.objects.get(pk=pk)
    except Dataset.DoesNotExist:
        return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

    file_obj = dataset.file
    file_obj.open('rb')
    file_bytes = file_obj.read()
    file_obj.close()

    header, rows = parse_rows(file_bytes, dataset.file.name)
    summary = dataset.summary_json or {}

    kpis = {
        'average_flowrate': summary.get('averages', {}).get('Flowrate'),
        'average_pressure': summary.get('averages', {}).get('Pressure'),
        'average_temperature': summary.get('averages', {}).get('Temperature'),
    }

    # Determine numeric columns for analytics
    numeric_cols = summary.get('numeric_columns') or [
        c for c in header if c not in ('Type', 'Equipment Name', 'Record')
    ]

    # Previous headers for schema drift (exclude current dataset)
    prev_headers = []
    for h in Dataset.objects.exclude(id=dataset.id).order_by('-created_at').values_list('summary_json', flat=True)[:4]:
        if isinstance(h, dict):
            prev_headers.append(h.get('all_columns'))
    flat_prev = [col for cols in prev_headers if cols for col in cols]

    quality = compute_quality(header, rows, previous_headers=flat_prev)
    correlations = compute_correlations(rows, numeric_cols)
    var_skew = compute_variance_skewness(rows, numeric_cols)
    clusters = kmeans_clusters(rows, [c for c in numeric_cols if c in ('Flowrate','Pressure','Temperature')])

    return Response({
        'header': header,
        'rows': rows,
        'kpis': kpis,
        'summary': summary,
        'data_quality': quality,
        'correlations': correlations,
        'variance_skewness': var_skew,
        'clustering': clusters,
    })
