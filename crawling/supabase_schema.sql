create table if not exists notices (
    id bigserial primary key,
    source text not null default 'hansung',
    notice_id text,
    title text not null,
    url text not null unique,
    posted_at date,
    posted_date_text text,
    category text,
    body text,
    views integer,
    raw jsonb not null default '{}'::jsonb,
    crawled_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_notices_source_posted_at
    on notices (source, posted_at desc);

create index if not exists idx_notices_category
    on notices (category);
