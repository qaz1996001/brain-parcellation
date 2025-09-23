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
-- Table structure for table `ksr_sum_st`
--

DROP TABLE IF EXISTS `ksr_sum_st`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ksr_sum_st` (
  `report_time` datetime NOT NULL,
  `cate` varchar(6) NOT NULL COMMENT 'category:\nRT : real-time\nYES : yesterday\nS1: day shift\nS2: night shift\nS3: over-night shift',
  `d_type` int(11) NOT NULL COMMENT 'd_type: \n1: area summary\n2: stage summary',
  `area_id` varchar(12) DEFAULT NULL,
  `area_name` varchar(24) DEFAULT NULL,
  `area_order` float DEFAULT NULL,
  `stage_id` varchar(16) NOT NULL DEFAULT '' COMMENT 'if d_type = 1 (show area), stage related fields is null',
  `stage_name` varchar(36) DEFAULT NULL,
  `stage_order` varchar(3) DEFAULT NULL,
  `demand` int(11) DEFAULT NULL,
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
  PRIMARY KEY (`report_time`,`cate`,`d_type`,`stage_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=1170 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ksr_sum_st`
--

LOCK TABLES `ksr_sum_st` WRITE;
/*!40000 ALTER TABLE `ksr_sum_st` DISABLE KEYS */;
INSERT INTO `ksr_sum_st` VALUES ('2018-05-04 01:59:07','RT',2,'C001-A','成型區-A',100,'Compaction','成型站','110',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','RT',2,'OIL01','真空含油區',500,'Oil_Impg','真空含油站','500',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','RT',2,'OQC-A','品檢區-A',600,'OQC','品檢站','600',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','RT',2,'PACK-A','包裝區',800,'Package','包裝站','800',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','RT',2,'S001','燒結區',200,'Sintering','燒結站','200',0,0,320,0,200,0,80,0,40,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','RT',2,'START-1','開始區',65,'STB','開始站','65',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','RT',2,'VIB01','震盪研磨區',420,'Vibr_Tumbling','震盪研磨站','420',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'C001-A','成型區-A',100,'Compaction','成型站','110',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'OIL01','真空含油區',500,'Oil_Impg','真空含油站','500',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'OQC-A','品檢區-A',600,'OQC','品檢站','600',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'PACK-A','包裝區',800,'Package','包裝站','800',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'S001','燒結區',200,'Sintering','燒結站','200',0,0,320,0,200,0,80,0,40,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'START-1','開始區',65,'STB','開始站','65',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318'),('2018-05-04 01:59:07','YES',2,'VIB01','震盪研磨區',420,'Vibr_Tumbling','震盪研磨站','420',0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,'2020-02-21 05:22:30.317318');
/*!40000 ALTER TABLE `ksr_sum_st` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:25
