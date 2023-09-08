-- filter by date
SELECT *
FROM taxi
WHERE
  timestamp > 1372636800 AND timestamp < 1372723199
  AND MISSING_DATA = FALSE;

SELECT *
FROM taxi
WHERE
  timestamp > 1372687200 AND timestamp < 1372687200 + 86399
  AND MISSING_DATA = FALSE;

SELECT *
FROM taxi
WHERE
  timestamp > 1372773600 AND timestamp < 1372773600 + 86399
  AND MISSING_DATA = FALSE;

SELECT *
FROM taxi
WHERE
  timestamp > 1372860000 AND timestamp < 1372860000 + 86399
  AND MISSING_DATA = FALSE;

SELECT *
FROM taxi
WHERE
  timestamp > 1372946400 AND timestamp < 1372946400 + 86399
  AND MISSING_DATA = FALSE;


-- Query range
SELECT trip_id
FROM taxi1
WHERE 
	ST_Contains('POLYGON((-8.59 41.15, -8.57 41.15, -8.57 41.17, -8.59 41.17, -8.59 41.15))'::geometry, ST_GeomFromText(linestring))
	AND linestring <> 'NULL';

SELECT trip_id
FROM taxi2
WHERE 
	ST_Contains('POLYGON((-8.45 41.15, -7.7 41.15, -7.7 41.5, -8.45 41.5, -8.45 41.15))'::geometry, ST_GeomFromText(linestring))
	AND linestring <> 'NULL';

SELECT trip_id
FROM taxi3
WHERE 
	ST_Contains('POLYGON((-8.6 41.2, -7.9 41.2, -7.9 41.7, -8.6 41.7, -8.6 41.2))'::geometry, ST_GeomFromText(linestring))
	AND linestring <> 'NULL';

SELECT trip_id
FROM taxi4
WHERE 
	ST_Contains('POLYGON((-8.8 41.2, -8 41.2, -8 42, -8.8 42, -8.8 41.2))'::geometry, ST_GeomFromText(linestring))
	AND linestring <> 'NULL';

SELECT trip_id
FROM taxi5
WHERE 
	ST_Contains('POLYGON((-8.6 41.2, -8 41.2, -8 41.5, -8.6 41.5, -8.6 41.2))'::geometry, ST_GeomFromText(linestring))
	AND linestring <> 'NULL';


-- Similar trajectories
SELECT levenshtein(to_string, '0.16,52|0.5,35|0.15,-18|0.3,-16'), trip_id
FROM taxi_sim1
WHERE length(to_string) < 255 
ORDER BY levenshtein(to_string, '0.16,52|0.5,35|0.15,-18|0.3,-16')
LIMIT 5;

SELECT levenshtein(to_string, '0.9,50|0.4,-50|0.5,-12|0.05,10'), trip_id
FROM taxi_sim2
WHERE length(to_string) < 255 
ORDER BY levenshtein(to_string, '0.9,50|0.4,-50|0.5,-12|0.05,10')
LIMIT 5;

SELECT levenshtein(to_string, '0.04,50|0.1,30|0.3,-30|0.2,52|0.5,-12|0.1,10|0.6,10'), trip_id
FROM taxi_sim3
WHERE length(to_string) < 255 
ORDER BY levenshtein(to_string, '0.04,50|0.1,30|0.3,-30|0.2,52|0.5,-12|0.1,10|0.6,10')
LIMIT 5;

SELECT levenshtein(to_string, '0.43,32|0.52,90|-0.2,-35'), trip_id
FROM taxi_sim4
WHERE length(to_string) < 255 
ORDER BY levenshtein(to_string, '0.43,32|0.52,90|-0.2,-35')
LIMIT 5;

SELECT levenshtein(to_string, '0.1,-10|0.4,32|0.2,-102|0.12,13|0.43,10'), trip_id
FROM taxi_sim5
WHERE length(to_string) < 255 
ORDER BY levenshtein(to_string, '0.1,-10|0.4,32|0.2,-102|0.12,13|0.43,10')
LIMIT 5;


-- knn
-- For graphing purposes.
(SELECT linestring::geography
FROM taxi1
WHERE linestring <> 'NULL'
ORDER BY linestring::geography <-> 'POINT(-8.613 41.145)'::geography
LIMIT 4)
UNION
(SELECT 'POINT(-8.613 41.145)'::geography);

SELECT trip_id, linestring::geography <-> 'POINT(-8.613 41.145)'::geography as dist
FROM taxi1
WHERE linestring <> 'NULL'
ORDER BY dist
LIMIT 4;

SELECT trip_id, linestring::geography <-> 'POINT(-8.2 41.3)'::geography as dist
FROM taxi2
WHERE linestring <> 'NULL'
ORDER BY dist
LIMIT 3;

SELECT trip_id, linestring::geography <-> 'POINT(-8.1 41.5)'::geography as dist
FROM taxi3
WHERE linestring <> 'NULL'
ORDER BY dist
LIMIT 2;

SELECT trip_id, linestring::geography <-> 'POINT(-8.4 41.7)'::geography as dist
FROM taxi4
WHERE linestring <> 'NULL'
ORDER BY dist
LIMIT 3;

SELECT trip_id, linestring::geography <-> 'POINT(-8.3 41.3)'::geography as dist
FROM taxi5
WHERE linestring <> 'NULL'
ORDER BY dist
LIMIT 3;


-- Skyline query
SELECT DISTINCT(p1.distance, p1.time)
FROM taxi_skyline1 p1
WHERE NOT EXISTS (
  SELECT 1
  FROM taxi_skyline1 p2
  WHERE 
    p2.distance <= p1.distance 
    AND p2.time <= p1.time 
    AND (p2.distance < p1.distance OR p2.time < p1.time)
	  OR (p1.distance is NULL or p1.time is NULL)
)

SELECT DISTINCT(p1.distance, p1.time)
FROM taxi_skyline2 p1
WHERE NOT EXISTS (
  SELECT 1
  FROM taxi_skyline2 p2
  WHERE 
    p2.distance <= p1.distance 
    AND p2.time <= p1.time 
    AND (p2.distance < p1.distance OR p2.time < p1.time)
	  OR (p1.distance is NULL or p1.time is NULL)
)

SELECT DISTINCT(p1.distance, p1.time)
FROM taxi_skyline3 p1
WHERE NOT EXISTS (
  SELECT 1
  FROM taxi_skyline3 p2
  WHERE 
    p2.distance <= p1.distance 
    AND p2.time <= p1.time 
    AND (p2.distance < p1.distance OR p2.time < p1.time)
	  OR (p1.distance is NULL or p1.time is NULL)
)

SELECT DISTINCT(p1.distance, p1.time)
FROM taxi_skyline4 p1
WHERE NOT EXISTS (
  SELECT 1
  FROM taxi_skyline4 p2
  WHERE 
    p2.distance <= p1.distance 
    AND p2.time <= p1.time 
    AND (p2.distance < p1.distance OR p2.time < p1.time)
	  OR (p1.distance is NULL or p1.time is NULL)
)

SELECT DISTINCT(p1.distance, p1.time)
FROM taxi_skyline5 p1
WHERE NOT EXISTS (
  SELECT 1
  FROM taxi_skyline51 p2
  WHERE 
    p2.distance <= p1.distance 
    AND p2.time <= p1.time 
    AND (p2.distance < p1.distance OR p2.time < p1.time)
	  OR (p1.distance is NULL or p1.time is NULL)
)

-- most_common_start
SELECT a.index AS point_a_id, b.index AS point_b_id
FROM taxi_start1 a
JOIN taxi_start1 b 
ON ST_Distance(a.point::geography, b.point::geography) < 0.0000001
  AND a.index < b.index
ORDER BY a.index, b.index;

SELECT a.index AS point_a_id, b.index AS point_b_id
FROM taxi_start2 a
JOIN taxi_start2 b 
ON ST_DWithin(a.point::geography, b.point::geography, 0.0000001) AND a.index < b.index
ORDER BY a.index, b.index;

SELECT a.index AS point_a_id, b.index AS point_b_id
FROM taxi_start3 a
JOIN taxi_start3 b 
ON ST_DWithin(a.point::geography, b.point::geography, 0.0000001) AND a.index < b.index
ORDER BY a.index, b.index;

SELECT a.index AS point_a_id, b.index AS point_b_id
FROM taxi_start4 a
JOIN taxi_start4 b 
ON ST_DWithin(a.point::geography, b.point::geography, 0.0000001) AND a.index < b.index
ORDER BY a.index, b.index;

SELECT a.index AS point_a_id, b.index AS point_b_id
FROM taxi_start5 a
JOIN taxi_start5 b 
ON ST_DWithin(a.point::geography, b.point::geography, 0.0000001) AND a.index < b.index
ORDER BY a.index, b.index;