from django.db import models, transaction
from django.utils import timezone
from django.contrib.auth.models import User
from django.core.validators import RegexValidator
from dateutil.relativedelta import relativedelta

# ---------- Choice Constants ----------
BOX_STATUS = [
    ('available', 'Available'),
    ('assigned', 'Assigned'),
    ('pending_recharge', 'Pending Recharge'),
    ('faulty', 'Faulty'),
    ('retired', 'Retired'),
    ('maintenance', 'Under Maintenance'),
]

CUSTOMER_STATUS = [
    ('new', 'New'),
    ('active', 'Active'),
    ('inactive', 'Inactive'),
]

PAYMENT_STATUS = [
    ('paid', 'Paid'),
    ('pending', 'Pending'),
    ('failed', 'Failed'),
]

PAYMENT_METHODS = [
    ('cash', 'Cash'),
    ('upi', 'UPI'),
    ('bank_transfer', 'Bank Transfer'),
    ('other', 'Other'),
]

SERVICE_TYPE = [
    ('new_connection', 'New Connection'),
    ('complaint', 'Complaint'),
    ('upgrade', 'Upgrade'),
    ('other', 'Other'),
]

SERVICE_STATUS = [
    ('open', 'Open'),
    ('assigned', 'Assigned'),
    ('in_progress', 'In Progress'),
    ('closed', 'Closed'),
]

BOX_ACTIONS = [
    ('assigned', 'Assigned'),
    ('unassigned', 'Unassigned'),
    ('recharged', 'Recharged'),
    ('maintenance', 'Maintenance'),
    ('repaired', 'Repaired'),
    ('retired', 'Retired'),
]


# ---------- Models ----------

class BoxType(models.Model):
    """Represents different types of boxes."""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name


class Plan(models.Model):
    """Represents an internet plan with pricing and validity."""
    name = models.CharField(max_length=100)
    box_type = models.ForeignKey(BoxType, on_delete=models.CASCADE, related_name='plans', null=True)
    speed = models.CharField(max_length=50)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    validity_days = models.PositiveIntegerField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='plans_created')
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def toggle_status(self):
        self.is_active = not self.is_active
        self.save(update_fields=['is_active'])

    def __str__(self):
        return f"{self.name} ({self.speed}) - ₹{self.price}"


class Box(models.Model):
    """Represents a physical box assigned to customers."""
    box_number = models.CharField(max_length=50, unique=True)
    box_type = models.ForeignKey(BoxType, on_delete=models.PROTECT)
    status = models.CharField(max_length=20, choices=BOX_STATUS, default='available')
    current_plan = models.ForeignKey(Plan, on_delete=models.SET_NULL, null=True, blank=True)
    plan_end_date = models.DateField(null=True, blank=True)
    remarks = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['box_number']
        indexes = [
            models.Index(fields=['box_number']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"{self.box_number} ({self.box_type.name})"

    def __repr__(self):
        return f"<Box {self.box_number}, Status: {self.status}>"

    @property
    def is_assigned(self):
        return self.box_assignment_related_id is not None

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
        if self.plan_end_date and self.is_assigned:
            return (self.plan_end_date - timezone.now().date()).days
        return None

    def assign_to_customer(self, customer, user):
        with transaction.atomic():
            if self.status != 'available':
                raise ValueError(f'Box {self.box_number} is not available (Status: {self.get_status_display()})')
            if customer.assigned_box:
                raise ValueError(f'Customer {customer.name} already has a box assigned.')
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
                raise ValueError(f'Box {self.box_number} is not currently assigned.')
            customer = self.box_assignment_related
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
            raise ValueError('Cannot recharge an unassigned box.')
        with transaction.atomic():
            today = timezone.now().date()
            self.plan_end_date = (
                self.plan_end_date + relativedelta(months=months)
                if self.plan_end_date and self.plan_end_date > today
                else today + relativedelta(months=months)
            )
            self.current_plan = plan
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
    """Represents a customer with contact and subscription details."""
    customer_id = models.CharField(max_length=10, unique=True, null=True, blank=True)
    name = models.CharField(max_length=100)
    contact = models.CharField(max_length=15, validators=[RegexValidator(regex=r'^\d{10}$', message='Enter a valid 10-digit phone number')])
    house_number = models.CharField(max_length=50)
    floor = models.CharField(max_length=10, blank=True)
    block_number = models.CharField(max_length=10, blank=True)
    area = models.CharField(max_length=100)
    city = models.CharField(max_length=50)
    pincode = models.CharField(max_length=6, validators=[RegexValidator(regex=r'^\d{6}$', message='Enter a valid 6-digit pincode')])
    status = models.CharField(max_length=10, choices=CUSTOMER_STATUS, default='new')
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
                prefix = 'CUS'
                last = Customer.objects.select_for_update().filter(customer_id__startswith=prefix).order_by('-customer_id').first()
                next_num = int(last.customer_id[len(prefix):]) + 1 if last else 1
                self.customer_id = f'{prefix}{next_num:06d}'

        today = timezone.now().date()
        if self.assigned_box and self.assigned_box.plan_end_date:
            self.status = 'active' if self.assigned_box.plan_end_date >= today else 'inactive'
        else:
            self.status = 'new'
        super().save(*args, **kwargs)


class Technician(models.Model):
    """Technician responsible for handling service requests."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=100, blank=True, null=True, default="Technician")
    phone = models.CharField(max_length=20, blank=True, null=True)
    area = models.CharField(max_length=100, blank=True)
    is_available = models.BooleanField(default=True)
    remarks = models.TextField(blank=True)

    def __str__(self):
        return self.name or "Unnamed Technician"


class ServiceRequest(models.Model):
    """Tracks support tickets and new connection requests."""
    ticket_id = models.CharField(max_length=10, unique=True, null=True, blank=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, null=True, blank=True)
    request_type = models.CharField(max_length=20, choices=SERVICE_TYPE)
    description = models.TextField()
    status = models.CharField(max_length=20, choices=SERVICE_STATUS, default='open')
    created_at = models.DateTimeField(auto_now_add=True)
    technician = models.ForeignKey(Technician, on_delete=models.SET_NULL, null=True, blank=True)
    assigned_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    new_connection_remarks = models.TextField(blank=True, null=True)
    new_connection_address = models.CharField(max_length=255, blank=True, null=True)
    new_connection_phone = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f"{self.ticket_id or 'Untitled'} - {self.get_request_type_display()}"

    def save(self, *args, **kwargs):
        if not self.ticket_id:
            with transaction.atomic():
                prefix = 'SRQ'
                last = ServiceRequest.objects.select_for_update().filter(ticket_id__startswith=prefix).order_by('-ticket_id').first()
                next_num = int(last.ticket_id[len(prefix):]) + 1 if last and last.ticket_id else 1
                self.ticket_id = f'{prefix}{next_num:06d}'
        super().save(*args, **kwargs)


class Payment(models.Model):
    """Stores customer payment details."""
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
    months_recharged = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.customer.name} - ₹{self.amount} ({self.get_status_display()})"

    @property
    def status_color(self):
        return {
            'paid': 'success',
            'pending': 'warning',
            'failed': 'danger',
        }.get(self.status, 'secondary')


class BoxHistory(models.Model):
    """Logs all changes to a Box: assignments, recharges, retirements, etc."""
    box = models.ForeignKey(Box, on_delete=models.CASCADE, related_name='histories')
    action = models.CharField(max_length=20, choices=BOX_ACTIONS)
    details = models.TextField(blank=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.box.box_number} - {self.get_action_display()} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
