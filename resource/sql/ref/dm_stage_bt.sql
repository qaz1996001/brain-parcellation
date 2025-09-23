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
-- Table structure for table `dm_stage_bt`
--

DROP TABLE IF EXISTS `dm_stage_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dm_stage_bt` (
  `area_id` varchar(12) NOT NULL,
  `stage_id` varchar(16) NOT NULL,
  `stage_name` varchar(36) DEFAULT NULL,
  `stage_order` varchar(3) NOT NULL COMMENT 'ope_no 的前三碼(main step, not sub steps)',
  `stage_desc` varchar(48) DEFAULT NULL COMMENT 'stage description',
  `ws_type` varchar(12) NOT NULL DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `rec_user` varchar(16) DEFAULT NULL COMMENT 'update user: log NT account',
  `rec_time` datetime DEFAULT NULL COMMENT 'record time',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`stage_id`,`area_id`,`stage_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=2048 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dm_stage_bt`
--

LOCK TABLES `dm_stage_bt` WRITE;
/*!40000 ALTER TABLE `dm_stage_bt` DISABLE KEYS */;
INSERT INTO `dm_stage_bt` VALUES ('C001-A','Forming','成型站','100','…','10','SYS','2020-03-17 15:51:36',NULL,'2020-03-17 07:51:36.808666'),('OIL01','Oil_Impg','真空含油站','500','…','50','SYS','2020-03-17 15:51:36',NULL,'2020-03-17 07:51:36.897082'),('OQC-A','OQC','品檢站','600','…','900','SYS','2020-03-17 15:51:36',NULL,'2020-04-13 01:12:19.059491'),('PACK-A','Package','包裝站','800','…','80','SYS','2020-03-17 15:51:37',NULL,'2020-03-17 07:51:37.032918'),('S001','Sintering','燒結站','200','…','20','SYS','2020-03-17 15:51:37',NULL,'2020-03-17 07:51:37.100055'),('SH01','Sizing','整形站','300','…','30','SYS','2020-03-17 15:51:37',NULL,'2020-03-17 07:51:37.166668'),('START-1','STB','開始站','065','…','6','SYS','2020-03-17 15:51:37',NULL,'2020-03-17 07:51:37.251747'),('VIB01','Vibr_Tumbling','震盪研磨站','420','…','42','SYS','2020-03-17 15:51:37',NULL,'2020-03-17 07:51:37.319974');
/*!40000 ALTER TABLE `dm_stage_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:23
