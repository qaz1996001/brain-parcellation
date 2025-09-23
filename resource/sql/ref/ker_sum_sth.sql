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
-- Table structure for table `ker_sum_sth`
--

DROP TABLE IF EXISTS `ker_sum_sth`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ker_sum_sth` (
  `report_time` datetime NOT NULL,
  `cate` varchar(6) NOT NULL COMMENT 'category:\nRT : real-time\nYES : yesterday\nS1: day shift\nS2: night shift\nS3: over-night shift',
  `area_id` varchar(12) DEFAULT NULL,
  `area_name` varchar(24) DEFAULT NULL,
  `area_order` int(11) DEFAULT NULL,
  `eqp_grp_id` varchar(12) NOT NULL,
  `eqp_grp` varchar(24) DEFAULT NULL,
  `eqp_grp_order` int(11) DEFAULT NULL,
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ker_sum_sth`
--

LOCK TABLES `ker_sum_sth` WRITE;
/*!40000 ALTER TABLE `ker_sum_sth` DISABLE KEYS */;
/*!40000 ALTER TABLE `ker_sum_sth` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:17:06
