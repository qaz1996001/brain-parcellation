CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ker_sum_st`
--

DROP TABLE IF EXISTS `ker_sum_st`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ker_sum_st` (
  `report_time` datetime NOT NULL,
  `cate` varchar(6) NOT NULL COMMENT 'category:\nRT : real-time\nYES : yesterday\nS1: day shift\nS2: night shift\nS3: over-night shift',
  `area_id` varchar(12) DEFAULT NULL,
  `area_name` varchar(24) DEFAULT NULL,
  `area_order` float DEFAULT NULL,
  `eqp_grp_id` varchar(12) NOT NULL,
  `eqp_grp` varchar(24) DEFAULT NULL,
  `eqp_grp_order` float DEFAULT NULL,
  `demand` int(11) DEFAULT NULL,
  `capacity` int(11) DEFAULT NULL,
  `avl` float DEFAULT NULL,
  `eff` float DEFAULT NULL,
  `lost` float DEFAULT NULL,
  `pwip` int(11) DEFAULT NULL,
  `wip` int(11) DEFAULT NULL,
  `move` int(11) DEFAULT NULL,
  `qwip` int(11) DEFAULT NULL,
  `qtime` float DEFAULT NULL,
  `rwip` int(11) DEFAULT NULL,
  `rtime` float DEFAULT NULL,
  `hwip` int(11) DEFAULT NULL,
  `htime` float DEFAULT NULL,
  `backup` int(11) DEFAULT NULL,
  `move_d` int(11) DEFAULT NULL,
  `move_n` int(11) DEFAULT NULL,
  `move_o` int(11) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`report_time`,`cate`,`eqp_grp_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=1024 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ker_sum_st`
--

LOCK TABLES `ker_sum_st` WRITE;
/*!40000 ALTER TABLE `ker_sum_st` DISABLE KEYS */;
INSERT INTO `ker_sum_st` VALUES ('2018-05-04 01:58:23','YES','C001-A','成型區-A',100,'CMPT-T1','Compaction-20T',100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','C001-A','成型區-A',100,'M-THK-1','MEA-THK',120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','C001-A','成型區-A',100,'M-WET-1','MEA-WG',110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','S001','燒結區',200,'SINTER-10','Sintering-1000',230,0,0,0,0,0,0,80,0,80,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','S001','燒結區',200,'SINTER-12','Sintering-1200',240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','S001','燒結區',200,'SINTER-6','Sintering-600',210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','S001','燒結區',200,'SINTER-8','Sintering-800',220,0,0,0,0,0,0,240,0,120,0,80,0,40,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-05-04 01:58:23','YES','START-1','開始區',65,'STB','WAFER_START',65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','C001-A','成型區-A',100,'CMPT-T1','Compaction-20T',100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','C001-A','成型區-A',100,'M-THK-1','MEA-THK',120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','C001-A','成型區-A',100,'M-WET-1','MEA-WG',110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','S001','燒結區',200,'SINTER-10','Sintering-1000',230,0,0,0,0,0,0,80,0,80,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','S001','燒結區',200,'SINTER-12','Sintering-1200',240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','S003','燒結區-3樓',203,'SINTER-3F','Sintering-3F',250,0,0,0,0,0,0,320,0,200,0,80,0,40,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','S001','燒結區',200,'SINTER-6','Sintering-600',210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','S001','燒結區',200,'SINTER-8','Sintering-800',220,0,0,0,0,0,0,240,0,120,0,80,0,40,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358'),('2018-11-07 11:30:20','RT','START-1','開始區',65,'STB','WAFER_START',65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:39.234358');
/*!40000 ALTER TABLE `ker_sum_st` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:32
