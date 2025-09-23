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
-- Table structure for table `oee_sum_tt`
--

DROP TABLE IF EXISTS `oee_sum_tt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `oee_sum_tt` (
  `report_time` datetime NOT NULL COMMENT '1. only keep 3 days record 2. tomorrow am 08:00 is real time data,others is history  -- 2017/06/26 lkchena...',
  `cate` varchar(6) NOT NULL COMMENT 'category:\r\nRT : real-time\r\nYES : yesterday\r\nS1: day shift\r\nS2: night shift\r\nS3: over-night shift',
  `tool_id` varchar(12) NOT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `device` varchar(8) NOT NULL,
  `avl` float(5,2) DEFAULT 0.00,
  `eff` float(5,2) DEFAULT 0.00,
  `lost` float(5,2) DEFAULT 0.00,
  `down` float(5,2) DEFAULT 0.00,
  `pm` float(5,2) DEFAULT 0.00,
  `hold` float(5,2) DEFAULT 0.00,
  `setup` float(5,2) DEFAULT 0.00 COMMENT 'mold setup time - 2019/02/02 lkchena',
  `test` float(5,2) DEFAULT 0.00,
  `mon` float(5,2) DEFAULT 0.00,
  `wait` float(5,2) DEFAULT 0.00,
  `off` float(5,2) DEFAULT 0.00,
  `move_act` float DEFAULT 0,
  `move_prod` float DEFAULT 0,
  `move_test` float DEFAULT 0 COMMENT 'test run before product 2019/02/04 lkchena',
  `move_eng` float DEFAULT 0,
  `move_avg` float DEFAULT 0,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`report_time`,`cate`,`tool_id`,`device`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC COMMENT='oee 需要向台機一樣, by hour snapshot record ??? 2017/07/19 lkchena';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `oee_sum_tt`
--

LOCK TABLES `oee_sum_tt` WRITE;
/*!40000 ALTER TABLE `oee_sum_tt` DISABLE KEYS */;
INSERT INTO `oee_sum_tt` VALUES ('2019-04-23 00:00:00','YES','AFMSX01','MAIN',100.00,0.14,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0,15,0,0,720,NULL,'2020-02-21 05:22:21.658465');
/*!40000 ALTER TABLE `oee_sum_tt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:19
