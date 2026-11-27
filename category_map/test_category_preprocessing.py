import pytest

from category_preprocessing import CATEGORIES, KEYWORDS, LISTING_TYPES, CONDITIONS, classify, clean, flatten_category_hierarchy

# (title, expected result)
CASES = [
    ('phone accessories',             'Electronics > Phones and Tablets'),
    ('laptop accessories',            'Electronics > Laptops and Computers'),
    ('camera accessories',            'Electronics > Photo and Video Equipment'),
    ('video game',                    'Electronics > Games and Consoles'),
    ('kids bicycle',                  'Baby and Kids > Toys'),
    ('baby monitor',                  'Baby and Kids > Kids Room'),
    ('baby bed',                      'Baby and Kids > Kids Room'),
    ('dress for sale',                "Clothing and Fashion > Women's Wardrobe"),
    ('office equipment',              'Business and Equipment > Equipment'),
    ('industrial tools',              'Business and Equipment > Equipment'),
    ('power tools',                   'Renovation and Construction > Tools'),
    ('restaurant equipment',          'Business and Equipment > Equipment'),
    ('medical equipment',             'Business and Equipment > Equipment'),
    ('photography services',          'Services > Events'),
    ('cartier wristwatch',            'Clothing and Fashion > Accessories'),
    ('Palm Pilot M515 PDA',           'Electronics > Everything Else'),
    ('toyota camry',                  'Vehicles > Cars'),
    ('golden retriever puppy',        'Pets and Animals > Pets'),
    ('iphone 13',                     'Electronics > Phones and Tablets'),
    ('Samsung Galaxy S21',            'Electronics > Phones and Tablets'),
    ('iPad Pro 11',                   'Electronics > Phones and Tablets'),
    ('apple watch',                   'Electronics > Phones and Tablets'),
    ('MacBook Air M1',                'Electronics > Laptops and Computers'),
    ('xbox game console',             'Electronics > Games and Consoles'),
    ('phone repair',                  'Services > Electronics'),
    ('Rolex Submariner',              'Clothing and Fashion > Accessories'),
    ('washing machine',               'Appliances > Clothing Care'),
    ('sewing machine',                'Appliances > Clothing Care'),
    ('corner sofa',                   'Home and Garden > Furniture'),
    ('dog food',                      'Pets and Animals > Products For Animals'),
    ('baby stroller',                 'Baby and Kids > Walking and Traveling'),
    ('Lego set',                      'Baby and Kids > Toys'),
    ('acoustic guitar',               'Hobbies and Sports > Music Instruments'),
    ('antique gramophone',            'Hobbies and Sports > Antiques and Art'),
    ('excavator',                     'Vehicles > Trucks and Special Machinery'),
    ('lawyer consultation',           'Services > Business Services'),
    ('fresh cheese',                  'Food and Beverages > Dairy and Eggs'),
    ('ceramic tile',                  'Renovation and Construction > Finishing Materials'),
    ('beehive with bees',             'Pets and Animals > Beekeeping'),
    ('Samsung refrigerator',          'Appliances > Kitchen Appliances'),
    ('Nikon D750 body',               'Electronics > Photo and Video Equipment'),
    ('Cat litter',                    'Pets and Animals > Products For Animals'),
    ('silver coins collection',       'Collectibles > Coins and Banknotes'),
    ('old postcards and stamps',      'Collectibles > Stamps and Postcards'),
    ('apartment for rent',            'Real Estate > Long Term Rentals'),
    ('apartment for sale in yerevan', 'Real Estate > For Sale'),
    ('daily rent apartment',          'Real Estate > Daily Rentals'),
    ('3 room apartment in yerevan',   'Real Estate'),
    # words that fit both Women's and Men's wardrobe -> main category only
    ('leather jacket',                'Clothing and Fashion'),
    ('Nike sneakers',                 'Clothing and Fashion'),
    # jobs
    ('we are hiring a sales manager', 'Jobs > Services Sector'),
    ('hiring accountant',             'Jobs > Business and Law'),
    ('my CV, looking for job',        'Jobs > Resumes'),
]


@pytest.mark.parametrize('title, expected', CASES)
def test_classification(title, expected):
    assert classify(title) == expected