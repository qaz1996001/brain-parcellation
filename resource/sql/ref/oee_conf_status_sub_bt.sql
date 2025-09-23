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
-- Table structure for table `oee_conf_status_sub_bt`
--

DROP TABLE IF EXISTS `oee_conf_status_sub_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `oee_conf_status_sub_bt` (
  `status` varchar(6) NOT NULL COMMENT 'tool status list: \nUP\nLOST\nDOWN\nPM\nHOLD\nTEST\nMON\nWAIT\nOFF\n\n\nHOLD-ENG\nHOLD-MFG\nAPM\nWMFG\nWCIM\nWENG',
  `status_sub` varchar(24) NOT NULL COMMENT 'for user key-in sub category 2019/12/08 lkchena',
  `sub_desc` varchar(32) DEFAULT NULL,
  `ord1` varchar(2) DEFAULT NULL COMMENT 'field order, let easy to read 2018/05/23 lkchena',
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`status`,`status_sub`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `oee_conf_status_sub_bt`
--

LOCK TABLES `oee_conf_status_sub_bt` WRITE;
/*!40000 ALTER TABLE `oee_conf_status_sub_bt` DISABLE KEYS */;
INSERT INTO `oee_conf_status_sub_bt` VALUES ('SETUP','ISSUE_MAN','人員因素','40',NULL,NULL,NULL,'2020-02-16 01:13:52.509321'),('SETUP','ISSUE_MOLD','模具因素','60',NULL,NULL,NULL,'2020-02-16 01:13:52.509321'),('SETUP','ISSUE_OTHERS','其他因素','90',NULL,NULL,NULL,'2020-02-16 01:13:52.509321'),('SETUP','ISSUE_POWDER','粉料異常','50',NULL,NULL,NULL,'2020-02-16 01:13:52.509321'),('SETUP','ISSUE_SPC','製品超差','20',NULL,NULL,NULL,'2020-02-16 01:13:52.509321'),('SETUP','ISSUE_TOOL','設備異常','30',NULL,NULL,NULL,'2020-02-16 01:13:52.509321'),('SETUP','NEW_ORDER','新製令單','10',NULL,NULL,NULL,'2020-02-16 01:13:52.509321');
/*!40000 ALTER TABLE `oee_conf_status_sub_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:29
