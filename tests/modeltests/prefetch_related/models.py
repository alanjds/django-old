from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes import generic
from django.db import models

## Basic tests

class Author(models.Model):
    name = models.CharField(max_length=50, unique=True)
    first_book = models.ForeignKey('Book', related_name='first_time_authors')
    favorite_authors = models.ManyToManyField(
        'self', through='FavoriteAuthors', symmetrical=False, related_name='favors_me')

    def __unicode__(self):
        return self.name

    class Meta:
        ordering = ['id']


class AuthorWithAge(Author):
    author = models.OneToOneField(Author, parent_link=True)
    age = models.IntegerField()


class FavoriteAuthors(models.Model):
    author = models.ForeignKey(Author, to_field='name', related_name='i_like')
    likes_author = models.ForeignKey(Author, to_field='name', related_name='likes_me')

    class Meta:
         ordering = ['id']


class AuthorAddress(models.Model):
    author = models.ForeignKey(Author, to_field='name', related_name='addresses')
    address = models.TextField()

    class Meta:
        ordering = ['id']

    def __unicode__(self):
        return self.address


class Book(models.Model):
    title = models.CharField(max_length=255)
    authors = models.ManyToManyField(Author, related_name='books')

    def __unicode__(self):
        return self.title

    class Meta:
        ordering = ['id']

class BookWithYear(Book):
    book = models.OneToOneField(Book, parent_link=True)
    published_year = models.IntegerField()
    aged_authors = models.ManyToManyField(
        AuthorWithAge, related_name='books_with_year')


class Reader(models.Model):
    name = models.CharField(max_length=50)
    books_read = models.ManyToManyField(Book, related_name='read_by')

    def __unicode__(self):
        return self.name

    class Meta:
        ordering = ['id']


## Models for default manager tests

class Qualification(models.Model):
    name = models.CharField(max_length=10)

    class Meta:
        ordering = ['id']


class TeacherManager(models.Manager):
    def get_query_set(self):
        return super(TeacherManager, self).get_query_set().prefetch_related('qualifications')


class Teacher(models.Model):
    name = models.CharField(max_length=50)
    qualifications = models.ManyToManyField(Qualification)

    objects = TeacherManager()

    def __unicode__(self):
        return "%s (%s)" % (self.name, ", ".join(q.name for q in self.qualifications.all()))

    class Meta:
        ordering = ['id']


class Department(models.Model):
    name = models.CharField(max_length=50)
    teachers = models.ManyToManyField(Teacher)

    class Meta:
        ordering = ['id']


## Generic relation tests

class TaggedItem(models.Model):
    tag = models.SlugField()
    content_type = models.ForeignKey(ContentType, related_name="taggeditem_set2")
    object_id = models.PositiveIntegerField()
    content_object = generic.GenericForeignKey('content_type', 'object_id')

    def __unicode__(self):
        return self.tag


class Bookmark(models.Model):
    url = models.URLField()
    tags = generic.GenericRelation(TaggedItem)
