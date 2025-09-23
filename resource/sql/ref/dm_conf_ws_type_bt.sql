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
-- Table structure for table `dm_conf_ws_type_bt`
--

DROP TABLE IF EXISTS `dm_conf_ws_type_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dm_conf_ws_type_bt` (
  `ws_type` varchar(12) NOT NULL DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `ws_func` varchar(24) DEFAULT NULL,
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`ws_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC COMMENT='0~99: not use\r\n100~799: others \r\n\n8xx: qc\n\r\n9xx: system use, ex: sintering one tool, but two omi -- 2020/03/18 lkchena';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dm_conf_ws_type_bt`
--

LOCK TABLES `dm_conf_ws_type_bt` WRITE;
/*!40000 ALTER TABLE `dm_conf_ws_type_bt` DISABLE KEYS */;
INSERT INTO `dm_conf_ws_type_bt` VALUES ('10','成形',NULL,'SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:26:50.213184'),('20','燒結','燒結頭尾設定在 tool 不是 stage','SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:55:45.694150'),('72','全檢',NULL,'SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:26:50.347697'),('900','品管',NULL,'SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:26:50.482873'),('98','重工',NULL,'SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:26:50.415491'),('xx_20','燒結-頭','燒結機台-前後各一台OMI','SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:55:45.694150'),('xx_21','燒結-尾','燒結機台-前後各一台OMI','SYS','2020/04/09 08:26:50',NULL,'2020-04-09 00:55:45.694651');
/*!40000 ALTER TABLE `dm_conf_ws_type_bt` ENABLE KEYS */;
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
