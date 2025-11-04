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
-- Table structure for table `base_conf_b_bt`
--

DROP TABLE IF EXISTS `base_conf_b_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `base_conf_b_bt` (
  `cate1` varchar(12) NOT NULL DEFAULT '*',
  `cate2` varchar(12) NOT NULL DEFAULT '*',
  `key_id` varchar(12) NOT NULL,
  `key_desc` varchar(32) DEFAULT NULL COMMENT 'description will show at title',
  `value1` varchar(24) DEFAULT NULL,
  `value2` varchar(24) DEFAULT NULL,
  `value3` varchar(24) DEFAULT NULL,
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(3) NULL DEFAULT current_timestamp(3) ON UPDATE current_timestamp(3),
  PRIMARY KEY (`cate1`,`cate2`,`key_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='different:\nbase_conf_bt: key data\nbase_conf_b_bt: non key data\n2020/01/24 lkchena';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `base_conf_b_bt`
--

LOCK TABLES `base_conf_b_bt` WRITE;
/*!40000 ALTER TABLE `base_conf_b_bt` DISABLE KEYS */;
INSERT INTO `base_conf_b_bt` VALUES ('DICT_EN_1','*','01','OK',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_EN_1','*','02','還機',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_EN_1','*','03','機台維修,預計      交回',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_EN_1','*','05','長官交代',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_EN_1','*','06','機台測試',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:23:37.614'),('DICT_EN_1','*','99','測試工程師字句',NULL,NULL,NULL,NULL,'SYS','2020/01/24 10:37:37',NULL,'2020-01-24 02:37:38.077'),('DICT_TO_1','*','01','OK',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_TO_1','*','02','換班',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-24 02:54:07.344'),('DICT_TO_1','*','03','機台故障,預計      交回',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_TO_1','*','04','休息時間',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_TO_1','*','05','長官交代',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:15:21.581'),('DICT_TO_1','*','06','測試',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-19 06:23:37.614');
/*!40000 ALTER TABLE `base_conf_b_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:30
