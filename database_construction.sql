CREATE TABLE IF NOT EXISTS public.taxi
(
    index integer,
    trip_id bigint NOT NULL,
    call_type character(1) COLLATE pg_catalog."default",
    origin_call character varying COLLATE pg_catalog."default",
    origin_stand character varying COLLATE pg_catalog."default",
    taxi_id integer,
    "timestamp" integer,
    day_type character(1) COLLATE pg_catalog."default",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi1
(
    index integer,
    trip_id bigint NOT NULL,
    call_type character(1) COLLATE pg_catalog."default",
    origin_call character varying COLLATE pg_catalog."default",
    origin_stand character varying COLLATE pg_catalog."default",
    taxi_id integer,
    "timestamp" integer,
    day_type character(1) COLLATE pg_catalog."default",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi2
(
    index integer,
    trip_id bigint NOT NULL,
    call_type character(1) COLLATE pg_catalog."default",
    origin_call character varying COLLATE pg_catalog."default",
    origin_stand character varying COLLATE pg_catalog."default",
    taxi_id integer,
    "timestamp" integer,
    day_type character(1) COLLATE pg_catalog."default",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi3
(
    index integer,
    trip_id bigint NOT NULL,
    call_type character(1) COLLATE pg_catalog."default",
    origin_call character varying COLLATE pg_catalog."default",
    origin_stand character varying COLLATE pg_catalog."default",
    taxi_id integer,
    "timestamp" integer,
    day_type character(1) COLLATE pg_catalog."default",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi4
(
    index integer,
    trip_id bigint NOT NULL,
    call_type character(1) COLLATE pg_catalog."default",
    origin_call character varying COLLATE pg_catalog."default",
    origin_stand character varying COLLATE pg_catalog."default",
    taxi_id integer,
    "timestamp" integer,
    day_type character(1) COLLATE pg_catalog."default",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default"
)


CREATE TABLE IF NOT EXISTS public.taxi5
(
    index integer,
    trip_id bigint NOT NULL,
    call_type character(1) COLLATE pg_catalog."default",
    origin_call character varying COLLATE pg_catalog."default",
    origin_stand character varying COLLATE pg_catalog."default",
    taxi_id integer,
    "timestamp" integer,
    day_type character(1) COLLATE pg_catalog."default",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_sim1
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    to_string character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_sim2
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    to_string character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_sim3
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    to_string character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_sim4
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    to_string character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_sim5
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    to_string character varying COLLATE pg_catalog."default"
)


CREATE TABLE IF NOT EXISTS public.taxi_skyline1
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    distance double precision,
    "time" integer
)

CREATE TABLE IF NOT EXISTS public.taxi_skyline2
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    distance double precision,
    "time" integer
)

CREATE TABLE IF NOT EXISTS public.taxi_skyline3
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    distance double precision,
    "time" integer
)

CREATE TABLE IF NOT EXISTS public.taxi_skyline4
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    distance double precision,
    "time" integer
)

CREATE TABLE IF NOT EXISTS public.taxi_skyline5
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    distance double precision,
    "time" integer
)


CREATE TABLE IF NOT EXISTS public.taxi_start1
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    point character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_start2
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    point character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_start3
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    point character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_start4
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    point character varying COLLATE pg_catalog."default"
)

CREATE TABLE IF NOT EXISTS public.taxi_start5
(
    index integer,
    useless integer,
    trip_id character varying COLLATE pg_catalog."default" NOT NULL,
    call_type "char",
    origin_call double precision,
    origin_stand double precision,
    taxi_id integer,
    "timestamp" integer,
    daytype "char",
    missing_data boolean,
    polyline character varying COLLATE pg_catalog."default",
    linestring character varying COLLATE pg_catalog."default",
    point character varying COLLATE pg_catalog."default"
)

-- The taxi table is the original database with all the dates.
-- Other table with a number at the end is the corrosponding date. e.g. taxi1 is the database
-- on July-01 etc...