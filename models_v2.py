from django.db import models, transaction
from django.utils import timezone
from django.conf import settings
from django.contrib.auth.models import User


class Plan(models.Model):
    name = models.CharField(max_length=100)
    box_type = models.ForeignKey('BoxType', on_delete=models.CASCADE, related_name='plans', null=True)
    speed = models.CharField(max_length=50)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    validity_days = models.IntegerField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    created_by = models.CharField(max_length=100)
    modified_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['name']  # Removed duplicate 'ordering'

    def toggle_status(self):
        self.is_active = not self.is_active
        self.save()

    def __str__(self):
        return f"{self.name} - ₹{self.price}"


class Box(models.Model):
    STATUS_CHOICES = [
        ('available', 'Available'),
        ('assigned', 'Assigned'),
        ('pending_recharge', 'Pending Recharge'),
        ('faulty', 'Faulty'),
        ('retired', 'Retired'),
        ('maintenance', 'Under Maintenance')
    ]
    box_number = models.CharField(max_length=50, unique=True)
    box_type = models.ForeignKey('BoxType', on_delete=models.PROTECT)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='available')
    current_plan = models.ForeignKey(Plan, on_delete=models.SET_NULL, null=True, blank=True)
    plan_end_date = models.DateField(null=True, blank=True)
    remarks = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['box_number']

    def __str__(self):
        return f"{self.box_number} ({self.box_type.name})"

    @property
    def is_assigned(self):
        return self.box_assignment is not None

    @property
    def is_due(self):
        today = timezone.now().date()
        return self.plan_end_date and self.is_assigned and self.plan_end_date <= today

    @property
    def is_due_today(self):
        today = timezone.now().date()
        return self.plan_end_date == today and self.is_assigned

    @property
    def is_past_due(self):
        today = timezone.now().date()
        return self.plan_end_date and self.is_assigned and self.plan_end_date < today

    @property
    def days_until_due(self):
        today = timezone.now().date()
        return (self.plan_end_date - today).days if self.plan_end_date and self.is_assigned else None

    @property
    def box_assignment(self):
        try:
            return self.box_assignment_related.get()
        except Customer.DoesNotExist:
            return None

    def assign_to_customer(self, customer, user):
        with transaction.atomic():
            if self.status != 'available':
                raise ValueError(f'Box {self.box_number} is not available (Status: {self.get_status_display()})')
            if customer.assigned_box:
                raise ValueError(f'Customer {customer.name} already has a box assigned')
            self.status = 'pending_recharge'
            customer.assigned_box = self
            customer.save()
            self.save()
            BoxHistory.objects.create(
                box=self,
                action='assigned',
                details=f'Assigned to {customer.name}',
                user=user
            )
            return True

    def unassign_from_customer(self, user, reason=''):
        with transaction.atomic():
            if not self.is_assigned:
                raise ValueError(f'Box {self.box_number} is not assigned')
            customer = self.box_assignment
            customer.assigned_box = None
            customer.save()
            self.status = 'available'
            self.current_plan = None
            self.plan_end_date = None
            self.save()
            BoxHistory.objects.create(
                box=self,
                action='unassigned',
                details=f'Unassigned from {customer.name}. Reason: {reason}',
                user=user
            )
            return True

    def recharge(self, plan, months, user):
        if not self.is_assigned:
            raise ValueError('Cannot recharge unassigned box')
        with transaction.atomic():
            self.current_plan = plan
            today = timezone.now().date()
            additional_days = months * 30
            if self.plan_end_date and self.plan_end_date > today:
                self.plan_end_date += timezone.timedelta(days=additional_days)
            else:
                self.plan_end_date = today + timezone.timedelta(days=additional_days)
            self.status = 'assigned'
            self.save()
            BoxHistory.objects.create(
                box=self,
                action='recharged',
                details=f'Recharged with {plan.name} for {months} months',
                user=user
            )
            return True


class Customer(models.Model):
    STATUS_CHOICES = [
        ('new', 'New'),
        ('active', 'Active'),
        ('inactive', 'Inactive')
    ]
    customer_id = models.CharField(max_length=10, unique=True, null=True, blank=True)
    name = models.CharField(max_length=100)
    contact = models.CharField(max_length=15)
    house_number = models.CharField(max_length=50)
    floor = models.CharField(max_length=10, blank=True)
    block_number = models.CharField(max_length=10, blank=True)
    area = models.CharField(max_length=100)
    city = models.CharField(max_length=50)
    pincode = models.CharField(max_length=6)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='new')
    due_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    last_payment_date = models.DateField(null=True, blank=True)
    remarks = models.TextField(blank=True)
    assigned_box = models.OneToOneField(Box, on_delete=models.SET_NULL, null=True, blank=True, related_name='box_assignment_related')

    class Meta:
        ordering = ['customer_id']

    def __str__(self):
        return f"{self.customer_id} - {self.name}"

    def save(self, *args, **kwargs):
        if not self.customer_id:
            with transaction.atomic():
                last_customer = Customer.objects.select_for_update().order_by('-customer_id').first()
                next_num = int(last_customer.customer_id[3:]) + 1 if last_customer and last_customer.customer_id else 1
                self.customer_id = f'CUS{next_num:06d}'
        today = timezone.now().date()
        if self.assigned_box:
            self.status = 'active' if self.assigned_box.plan_end_date and self.assigned_box.plan_end_date >= today else 'inactive'
        else:
            self.status = 'new'
        super().save(*args, **kwargs)


class Technician(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=100, blank=True, null=True, default="Technician")
    phone = models.CharField(max_length=20, blank=True, null=True)
    area = models.CharField(max_length=100, blank=True)
    is_available = models.BooleanField(default=True)
    remarks = models.TextField(blank=True)

    def __str__(self):
        return self.name or "Unnamed Technician"


class ServiceRequest(models.Model):
    TYPE_CHOICES = [
        ('new_connection', 'New Connection'),
        ('complaint', 'Complaint'),
        ('upgrade', 'Upgrade'),
        ('other', 'Other'),
    ]
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('assigned', 'Assigned'),
        ('in_progress', 'In Progress'),
        ('closed', 'Closed'),
    ]
    ticket_id = models.CharField(max_length=10, unique=True, null=True, blank=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, null=True, blank=True)
    request_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    description = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    created_at = models.DateTimeField(auto_now_add=True)
    technician = models.ForeignKey(Technician, on_delete=models.SET_NULL, null=True, blank=True)
    assigned_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    new_connection_remarks = models.TextField(blank=True, null=True)
    new_connection_address = models.CharField(max_length=255, blank=True, null=True)
    new_connection_phone = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f"{self.ticket_id or 'Untitled'} - {self.get_request_type_display()}"


class Payment(models.Model):
    PAYMENT_STATUS = [
        ('paid', 'Paid'),
        ('pending', 'Pending'),
        ('failed', 'Failed')
    ]
    PAYMENT_METHODS = [
        ('cash', 'Cash'),
        ('upi', 'UPI'),
        ('bank_transfer', 'Bank Transfer'),
        ('other', 'Other')
    ]
    box = models.ForeignKey(Box, on_delete=models.CASCADE)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    plan = models.ForeignKey(Plan, on_delete=models.SET_NULL, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateTimeField(auto_now_add=True)
    payment_method = models.CharField(max_length=20, choices=PAYMENT_METHODS, default='cash')
    status = models.CharField(max_length=20, choices=PAYMENT_STATUS, default='pending')
    received_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='payments_received')
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='payments_updated')
    remarks = models.TextField(blank=True)
    transaction_id = models.CharField(max_length=100, blank=True)
    months_recharged = models.IntegerField(default=1)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.customer.name} - ₹{self.amount} ({self.get_status_display()})"

    @property
    def status_color(self):
        return {
            'paid': 'success',
            'pending': 'warning',
            'failed': 'danger'
        }.get(self.status, 'secondary')


class BoxHistory(models.Model):
    ACTION_CHOICES = [
        ('assigned', 'Assigned'),
        ('unassigned', 'Unassigned'),
        ('recharged', 'Recharged'),
        ('maintenance', 'Maintenance'),
        ('repaired', 'Repaired'),
        ('retired', 'Retired'),
    ]
    box = models.ForeignKey(Box, on_delete=models.CASCADE)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    details = models.TextField(blank=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.box.box_number} - {self.get_action_display()} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
