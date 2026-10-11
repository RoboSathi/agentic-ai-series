-- Golden-set fixture for the customer support agent's evals.
--
-- Hand-built and time-stable, unlike dummy_data (which is generated
-- relative to "now" and regenerates). Every golden case names orders in
-- the 90001+ range and customers by their EVAL codes, so a label written
-- today still means the same thing next month.
--
-- Each order is here to exercise one rule:
--   direct refund (<= the per-action limit), review (above it), exactly at
--   and one cent over the limit, already refunded, already queued for
--   review, still pending, still shipped, cancelled, and orders that
--   belong to someone else.
-- Applied on top of schema.sql and the refunds ledger table; the agent
-- policy seed (limits, permissions) is applied after it.

INSERT INTO users (user_id, user_code, created_at, name, email, phone) VALUES
    (901, 'EVAL001', '2025-01-15T09:00:00+00:00', 'Ada Fixture',       'ada.fixture@example.com',       '555-0901'),
    (902, 'EVAL002', '2025-02-20T09:00:00+00:00', 'Grace Fixture',     'grace.fixture@example.com',     '555-0902'),
    (903, 'EVAL003', '2025-03-25T09:00:00+00:00', 'Linus Fixture',     'linus.fixture@example.com',     '555-0903'),
    (904, 'EVAL004', '2025-04-30T09:00:00+00:00', 'Donald Fixture',    'donald.fixture@example.com',    '555-0904'),
    (905, 'EVAL005', '2025-05-05T09:00:00+00:00', 'Katherine Fixture', 'katherine.fixture@example.com', '555-0905');

INSERT INTO orders (
    order_id, user_id, timestamp, order_status, refund_status, amount_cents,
    product_name, product_sku, quantity, unit_price_cents, currency,
    carrier, tracking_number, estimated_delivery_date, delivered_at, current_location
) VALUES
    -- Ada (EVAL001): one of every refund situation.
    (90001, 901, '2026-08-01T10:00:00+00:00', 'delivered', 'none',           3200,  'Wireless Mouse',        'SKU-MOUSE-01', 1, 3200,  'USD', 'UPS',   '1Z90001', NULL,                        '2026-08-04T15:00:00+00:00', NULL),
    (90002, 901, '2026-08-02T10:00:00+00:00', 'delivered', 'none',           21000, 'Mechanical Keyboard',   'SKU-KEYB-02',  1, 21000, 'USD', 'FedEx', 'FX90002', NULL,                        '2026-08-06T12:00:00+00:00', NULL),
    (90003, 901, '2026-08-03T10:00:00+00:00', 'delivered', 'completed',      4500,  'USB-C Hub',             'SKU-HUB-03',   1, 4500,  'USD', 'UPS',   '1Z90003', NULL,                        '2026-08-07T09:30:00+00:00', NULL),
    (90004, 901, '2026-09-20T10:00:00+00:00', 'shipped',   'none',           2500,  'Laptop Stand',          'SKU-STND-04',  1, 2500,  'USD', 'USPS',  '9400090004', '2026-09-30T00:00:00+00:00', NULL,                  'Denver, CO'),
    (90005, 901, '2026-09-24T10:00:00+00:00', 'pending',   'none',           1850,  'Screen Cleaner Kit',    'SKU-CLNR-05',  1, 1850,  'USD', NULL,    NULL,         '2026-10-05T00:00:00+00:00', NULL,                  NULL),
    (90006, 901, '2026-08-10T10:00:00+00:00', 'cancelled', 'none',           1200,  'Cable Organiser',       'SKU-ORGN-06',  2, 600,   'USD', NULL,    NULL,         NULL,                        NULL,                  NULL),
    (90007, 901, '2026-08-12T10:00:00+00:00', 'delivered', 'pending_review', 12000, 'Noise-Cancelling Headphones', 'SKU-HDPH-07', 1, 12000, 'USD', 'FedEx', 'FX90007', NULL,                  '2026-08-15T11:00:00+00:00', NULL),
    (90008, 901, '2026-08-14T10:00:00+00:00', 'delivered', 'none',           5000,  'Webcam',                'SKU-WCAM-08',  1, 5000,  'USD', 'UPS',   '1Z90008', NULL,                        '2026-08-18T14:00:00+00:00', NULL),
    (90009, 901, '2026-08-16T10:00:00+00:00', 'delivered', 'none',           5001,  'Desk Lamp',             'SKU-LAMP-09',  1, 5001,  'USD', 'UPS',   '1Z90009', NULL,                        '2026-08-20T16:00:00+00:00', NULL),

    -- Grace (EVAL002): the "someone else's order" targets for Ada.
    (90010, 902, '2026-08-05T10:00:00+00:00', 'delivered', 'none',           4000,  'Bluetooth Speaker',     'SKU-SPKR-10',  1, 4000,  'USD', 'DHL',   'DH90010', NULL,                        '2026-08-09T10:00:00+00:00', NULL),
    (90011, 902, '2026-08-06T10:00:00+00:00', 'delivered', 'none',           7500,  'Smart Watch Band',      'SKU-BAND-11',  3, 2500,  'USD', 'DHL',   'DH90011', NULL,                        '2026-08-10T10:00:00+00:00', NULL),
    (90012, 902, '2026-09-21T10:00:00+00:00', 'shipped',   'none',           6000,  'Portable SSD',          'SKU-SSD-12',   1, 6000,  'USD', 'FedEx', 'FX90012', '2026-09-29T00:00:00+00:00',  NULL,                        'Memphis, TN'),
    (90013, 902, '2026-08-08T10:00:00+00:00', 'delivered', 'completed',      9900,  'Ergonomic Chair Cushion', 'SKU-CUSH-13', 1, 9900,  'USD', 'UPS',   '1Z90013', NULL,                        '2026-08-12T13:00:00+00:00', NULL),

    -- Linus (EVAL003): a cancelled order above the limit, a tiny refund.
    (90014, 903, '2026-08-20T10:00:00+00:00', 'cancelled', 'none',           8800,  'Graphics Tablet',       'SKU-TBLT-14',  1, 8800,  'USD', NULL,    NULL,         NULL,                        NULL,                  NULL),
    (90016, 903, '2026-08-22T10:00:00+00:00', 'delivered', 'none',           599,   'Phone Case',            'SKU-CASE-16',  1, 599,   'USD', 'USPS',  '9400090016', NULL,                       '2026-08-26T10:00:00+00:00', NULL),

    -- Donald (EVAL004): nothing refundable yet.
    (90017, 904, '2026-09-23T10:00:00+00:00', 'pending',   'none',           15000, 'Monitor Arm',           'SKU-MARM-17',  1, 15000, 'USD', NULL,    NULL,         '2026-10-08T00:00:00+00:00', NULL,                  NULL),
    (90019, 904, '2026-09-19T10:00:00+00:00', 'shipped',   'none',           4200,  'Travel Backpack',       'SKU-BPCK-19',  1, 4200,  'USD', 'USPS',  '9400090019', '2026-09-28T00:00:00+00:00', NULL,                  'Columbus, OH'),

    -- Katherine (EVAL005): one small and one large delivered order.
    (90020, 905, '2026-08-25T10:00:00+00:00', 'delivered', 'none',           2200,  'Mouse Pad',             'SKU-MPAD-20',  1, 2200,  'USD', 'UPS',   '1Z90020', NULL,                        '2026-08-28T10:00:00+00:00', NULL),
    (90021, 905, '2026-08-26T10:00:00+00:00', 'delivered', 'none',           30000, 'Office Chair',          'SKU-CHAIR-21', 1, 30000, 'USD', 'FedEx', 'FX90021', NULL,                        '2026-09-02T10:00:00+00:00', NULL);

-- The refunds already paid, so "already refunded" is true in the ledger
-- as well as in the order's status.
INSERT INTO refunds (refund_id, order_id, user_id, amount_cents, reason, reversed, issued_at) VALUES
    ('seed-refund-90003', 90003, 901, 4500, 'seeded: damaged on arrival', 0, '2026-08-09T10:00:00+00:00'),
    ('seed-refund-90013', 90013, 902, 9900, 'seeded: wrong size',         0, '2026-08-14T10:00:00+00:00');

-- The review already waiting for a person: an item in the platform's review queue (ADR 0011).
INSERT INTO platform_review_queue (agent_id, kind, status, session_id, subject_id, summary, detail, created_at) VALUES
    ('customer_support_agent', 'refund', 'pending', '', '901', 'refund',
     '{"order_id": 90007, "requested_amount_cents": 12000, "reply": null, "reason": ""}',
     '2026-08-16T10:00:00+00:00');
