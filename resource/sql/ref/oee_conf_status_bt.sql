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
-- Table structure for table `oee_conf_status_bt`
--

DROP TABLE IF EXISTS `oee_conf_status_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `oee_conf_status_bt` (
  `status` varchar(6) NOT NULL COMMENT 'tool status list: \nUP\nLOST\nDOWN\nPM\nHOLD\nTEST\nMON\nWAIT\nOFF\n\n\nHOLD-ENG\nHOLD-MFG\nAPM\nWMFG\nWCIM\nWENG',
  `desc` varchar(32) DEFAULT NULL,
  `ord1` varchar(2) DEFAULT NULL COMMENT 'field order, let easy to read 2018/05/23 lkchena',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=1820 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `oee_conf_status_bt`
--

LOCK TABLES `oee_conf_status_bt` WRITE;
/*!40000 ALTER TABLE `oee_conf_status_bt` DISABLE KEYS */;
INSERT INTO `oee_conf_status_bt` VALUES ('DOWN','故障','50',NULL,'2020-02-16 01:13:51.925628'),('HOLD','暫置,不可生產','55',NULL,'2020-02-16 01:13:51.925628'),('LOST','機台正常,閒置中','30',NULL,'2020-02-16 01:13:51.925628'),('OFF','關機','95',NULL,'2020-02-16 01:13:51.925628'),('OPT','使用者定義','79',NULL,'2020-03-17 04:29:23.874204'),('PM','機台保養','60',NULL,'2020-02-16 01:13:51.925628'),('SETUP','裝模調機','20',NULL,'2020-02-16 01:13:51.925628'),('TEST','測試 / 試打','25',NULL,'2020-02-16 01:13:51.925628'),('UP','正常生產','10',NULL,'2020-02-16 01:13:51.925628'),('WAIT','等待處理','70',NULL,'2020-02-16 01:13:51.925628');
/*!40000 ALTER TABLE `oee_conf_status_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:35
